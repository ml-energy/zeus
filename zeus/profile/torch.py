# Copyright (C) 2022 Jae-Won Chung <jwnchung@umich.edu>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the ProfileDataLoader class."""

from __future__ import annotations

import atexit
import os
import subprocess
import time
from typing import Literal

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from zeus.util import get_env


class ProfileDataLoader(DataLoader):
    """A DataLoader class that profiles power and time.

    `ProfileDataLoader` acts just like an ordinary
    [`DataLoader`][torch.utils.data.DataLoader] while profiling power
    consumption and epoch latency under the hood. Power profiling is done by
    spawning the Zeus power monitor as a subprocess. The latency of each epoch
    will be printed out to stdout.

    `ProfileDataLoader` interfaces with the outside world with environment variables.

    - `ZEUS_LOG_PREFIX`      : Prefix for power and time log files.
                               Power log: `f"{log_prefix}+gpu{index}.power.csv"`
                               Time log : `f"{log_prefix}.time.csv"`
    - `ZEUS_MONITOR_PATH`    : Path to the Zeus power monitor.
                               (Default: `"/workspace/zeus/zeus_monitor/zeus_monitor"`)
    - `ZEUS_MONITOR_SLEEP_MS`: How many milliseconds to sleep after measuring power.
                               This is passed to the monitor. (Default: `"100"`)

    `ProfileDataLoader` supports training on only a subset of the dataset and
    scaling time measurements as if trained on the entire dataset.

    `ProfileDataLoader` will assume that training is happening on all GPUs visible
    and spawn one Zeus power monitor process for each GPU. If this is not what you
    want, set `CUDA_VISIBLE_DEVICES` or spawn a Docker container that only mounts
    the GPUs that you would like to use.
    """

    # Power monitor processes
    monitor: list[subprocess.Popen] | None = None

    def __init__(
        self,
        *args,
        batch_size: int,
        split: Literal["train", "eval"],
        subset_proportion: float = 1.0,
        eat_batch_size: bool = False,
        only_scale_time: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the ProfileDataLoader.

        Args:
            batch_size: Batch size to use.
            split: Dataset split. Used when printing out epoch latency.
            subset_proportion: Should be between 0.0 and 1.0. When specified,
                only that proportion of the dataset will be used and the dataloader will
                stop early. Then, the measured epoch latency will be scaled as if the
                whole datset was used.
            only_scale_time: If True, the whole dataset will be used for training, but
                the measured epoch latency will still be scaled based on the value of
                `subset_proportion`. This is useful when you already manually scaled down
                the size of an existing dataset, but still want to simulate training the
                original large dataset.
            eat_batch_size: If True, does not pass the `batch_size` argument to the
                constructor of DataLoader. You won't usually need this.
        """
        # Assumes one epoch per invocation of __iter__.
        self.epoch = 0
        if split not in ["train", "eval"]:
            raise ValueError("split should be either 'train' or 'eval'.")
        self.split = split
        self.scaling_factor = 1.0
        self.start1 = None
        self.start2 = None

        # Retrieve environment variables needed.
        self.monitor_path = get_env("ZEUS_MONITOR_PATH", str)
        self.monitor_sleep_ms = get_env("ZEUS_MONITOR_SLEEP_MS", int, default=100)
        self.log_prefix = get_env("ZEUS_LOG_PREFIX", str)

        # Check if the Zeus power monitor is executable.
        if not os.access(self.monitor_path, os.X_OK):
            raise RuntimeError(f"'{self.monitor_path}' is not executable")

        # Create time.csv and write header.
        if self.split == "train":
            ProfileDataLoader.time_file = open(self.log_prefix + ".time.csv", "w")
            self.time_file.write("epoch,split,time\n")
            self.time_file.flush()

        # Slice out subset of dataset if subset_proportion is given.
        dataset = kwargs["dataset"] if "dataset" in kwargs else args[0]
        if subset_proportion > 1.0 or subset_proportion <= 0.0:
            raise ValueError("subset_proportion should be > 0.0 and <= 1.0.")
        if subset_proportion < 1.0:
            subset_indices = list(range(0, len(dataset), round(1 / subset_proportion)))  # type: ignore
            # See note in __next__ for more about scaling.
            self.scaling_factor = len(dataset) / (len(subset_indices) - batch_size)
            if not only_scale_time:
                subset = Subset(dataset, subset_indices)
                if "dataset" in kwargs:
                    kwargs["dataset"] = subset
                else:
                    args = (subset, *args[1:])

        # Call the constructor of DataLoader.
        if eat_batch_size:
            super().__init__(*args, **kwargs)  # type: ignore
        else:
            super().__init__(*args, batch_size=batch_size, **kwargs)  # type: ignore

    def __iter__(self):
        """Wrap the original `__iter__`, but with power profiling."""
        # On the first epoch, start the Zeus power monitors for each GPU.
        if ProfileDataLoader.monitor is None:
            if (count := torch.cuda.device_count()) == 1 or dist.get_rank() == 0:
                ProfileDataLoader.monitor = []
                for index in range(count):
                    monitor_cmd = [
                        self.monitor_path,
                        self.log_prefix + f"+gpu{index}.power.csv",  # Power log file
                        "0",  # Duration
                        str(self.monitor_sleep_ms),  # Monitor sleep time (ms)
                        str(index),  # GPU_id
                    ]
                    print(f"Launching Zeus monitor {index}...")
                    ProfileDataLoader.monitor.append(
                        subprocess.Popen(
                            monitor_cmd,
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )
                    )
                atexit.register(kill_monitor)
        # pylint: disable=attribute-defined-outside-init
        self.iter = super().__iter__()
        self.epoch += 1
        self.start1 = None
        self.start2 = None
        return self

    def __next__(self):
        """Wrap the original `__next__`, but with power profiling."""
        try:
            # Special treatment for the first batch.
            # Data loading takes significantly more time for the first batch. Thus, if we
            # simply measure the first ~ last batch latency of the subset of the dataset
            # and multiply 1/subset_proportion, we end up overestimating time_per_epoch.
            # Thus, we isolate the processing time of the first batch (start2 - start1),
            # scale up second ~ last batch latency with the adjusted scaling factor, and
            # later add the processing time of the first batch.
            #
            # Strange if nest to make the common case number of if statement executions 1.
            if self.start2 is None:
                if self.start1 is None:
                    self.start1 = time.time()
                else:
                    self.start2 = time.time()
            return self.iter.__next__()
        except StopIteration:
            end = time.time()
            if self.start1 and self.start2:
                scaled_time = (
                    self.scaling_factor * (end - self.start2)
                    + self.start2
                    - self.start1
                )
                if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                    self.time_file.write(f"{self.epoch},{self.split},{scaled_time}\n")
                    self.time_file.flush()
                    print(
                        f"epoch {self.epoch} {self.split} time consumed: {scaled_time:.2f}s"
                    )
            raise


def kill_monitor():
    """Kill all Zeus power monitors."""
    monitor = ProfileDataLoader.monitor
    if monitor is not None:
        for i, proc in enumerate(monitor):
            proc.kill()
            print(f"Stopped Zeus monitor {i}.")
