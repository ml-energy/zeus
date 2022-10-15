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

"""Defines the ZeusDataLoader class."""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import time
import logging
from functools import cached_property
from pathlib import Path
from typing import Generator, Literal
import numpy as np

import pynvml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from zeus import analyze
from zeus.util.check import get_env
from zeus.util.metric import ZeusCostThresholdExceededException, zeus_cost

# JIT profiling states
NOT_PROFILING = "NOT_PROFILING"
WARMING_UP = "WARMING_UP"
PROFILING = "PROFILING"

# Config logging
LOG = logging.Logger(__name__)
LOG.setLevel(logging.INFO)
LOG_HANDLER = logging.StreamHandler()
LOG_FORMATTER = logging.Formatter("%(asctime)s %(message)s")
LOG_HANDLER.setFormatter(LOG_FORMATTER)
LOG.addHandler(LOG_HANDLER)


class ZeusDataLoader(DataLoader):
    r"""Profiles and optimizes GPU power limit.

    `ZeusDataLoader` is integrated into the DNN training script, and transparently
    profiles power and time consumption to determine the optimal GPU power limit.

    # Integration examples

    ## Single-GPU

    ```python
    from zeus.run import ZeusDataLoader

    # The one instantiated with max_epochs becomes the train dataloader
    train_loader = ZeusDataLoader(train_set, batch_size=256, max_epochs=100)
    eval_loader = ZeusDataLoader(eval_set, batch_size=256)

    for epoch_number in train_loader.epochs():
        for batch in train_loader:
            # Learn from batch
        for batch in eval_loader:
            # Evaluate on batch

        train_loader.report_metric(validation_metric)
    ```

    ## Data parallel with multi-GPU on a single-node

    !!! Important
        Zeus assumes that exactly one process manages one GPU, and hence
        one instance of [`ZeusDataLoader`][zeus.run.ZeusDataLoader] exists
        for each GPU.

    Users can integrate Zeus into existing data parallel training scripts
    with five specific steps, which are noted below in the comments.

    Please refer to
    [our integration example with ImageNet](https://github.com/SymbioticLab/Zeus/tree/master/examples/imagenet/train.py)
    for a complete example.

    ```python
    import torch
    import torch.distributed as dist
    import torchvision

    from zeus.run import ZeusDataLoader

    # Step 1: Initialize the default process group.
    # This should be done before instantiating `ZeusDataLoader`.
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
    )

    # Step 2: Create a model and wrap it with `DistributedDataParallel`.
    model = torchvision.models.resnet18()
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    # Zeus assumes that exactly one process manages one GPU. If you are doing data
    # parallel training, please use `DistributedDataParallel` for model replication
    # and specify the `device_ids` and `output_device` as below:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    # Step 3: Create instances of `DistributedSampler` to partition the dataset
    # across the GPUs.
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_set)

    # Step 4: Instantiate `ZeusDataLoader`.
    # `distributed="dp"` tells `ZeusDataLoader` to operate in data parallel mode.
    # The one instantiated with `max_epochs` becomes the train dataloader.
    train_loader = ZeusDataLoader(train_set, batch_size=256, max_epochs=100,
                                  sampler=train_sampler, distributed="dp")
    eval_loader = ZeusDataLoader(eval_set, batch_size=256, sampler=eval_sampler,
                                 distributed="dp")

    # Step 5: Training loop.
    # Use the train dataloader's `epochs` generator to allow Zeus to early-stop
    # based on the cost. Use `report_metric` to let Zeus know the current
    # validation metric.
    for epoch_number in train_loader.epochs():
        for batch in train_loader:
            # Learn from batch
        for batch in eval_loader:
            # Evaluate on batch

        # Make sure you all-reduce the validation metric across all GPUs,
        # since Zeus expects the final validation metric.
        val_metric_tensor = torch.tensor([validation_metric], device="cuda")
        dist.all_reduce(val_metric_tensor, async_op=False)
        train_loader.report_metric(val_metric_tensor.item())
    ```

    # Environment variables

    `ZeusDataLoader` interfaces with the outside world via environment variables.
    Thus, while `ZeusDataLoader` is paired together with
    [`ZeusMaster`][zeus.run.ZeusMaster] in example scripts, any other "driver" can
    use `ZeusDataLoader` as long as it sets appropriate environment variables.

      - `ZEUS_TARGET_METRIC` : Required. Zeus will stop training when this target
                               validation metric is reached. Will be cast to float.
      - `ZEUS_LOG_DIR`       : Directory to store profiling logs. (Default:` "zeus_log"`)
      - `ZEUS_JOB_ID`        : String to prefix in logs. (Default:` "zeus"`)
      - `ZEUS_COST_THRESH`   : Stop training when the energy-time cost will exceed
                               this threshold.  (Default:` "inf"`)
      - `ZEUS_ETA_KNOB`      : $\eta$ knob to tradeoff between energy and time.
                               Larger values reduce more energy and sacrifice time.
                               (Default:` "0.5"`)
      - `ZEUS_MONITOR_PATH`  : Path to the Zeus power monitor binary.
                               (Default:` "/workspace/zeus/zeus_monitor/zeus_monitor"`)
      - `ZEUS_PROFILE_PARAMS`: Warmup and measure iterations for each power limit,
                               separated by a comma. (Default:` "10,40"`)
      - `ZEUS_USE_OPTIMAL_PL`: Whether to actually use the optimal power limit found.
                               Setting this to false is the Observer Mode described
                               in Section 5. (Default:` "True"`)
    """

    # Global power monitor instances.
    monitors: list[subprocess.Popen] = []

    # The power limit currently set for the GPU.
    current_gpu_pl: int = 0

    # Train batch size to be accessed by the eval dataloader.
    train_batch_size: int = 0

    # Length of the eval dataloader. `epochs` in the train dataloader needs this.
    eval_num_samples: int = 0

    # Train-time power profiling result. Maps power limit to avg_power & throughput.
    train_power_result: dict[int, float] = {}
    train_tput_result: dict[int, float] = {}

    # Eval-time power profiling result. Maps power limit to avg_power & throughput.
    eval_power_result: dict[int, float] = {}
    eval_tput_result: dict[int, float] = {}

    # Cost-optimal power limit. Set by the train dataloader after the last power limit
    # was explored.
    optimal_pl: int = 0

    # Train epoch measurements for time/energy accounting.
    train_epoch_time: list[float] = []
    # The master process will record ALL GPUs' energy consumption during training.
    # GPU_i's energy records is `train_epoch_energy[i]`.
    train_epoch_energy: np.ndarray = np.empty(0)

    # Eval-time latency profiling result. Maps power limit to epoch latency.
    eval_epoch_time: list[float] = []
    # The master process will record ALL GPUs' energy consumption during evaluation.
    # GPU_i's energy records is `eval_epoch_energy[i]`.
    eval_epoch_energy: np.ndarray = np.empty(0)

    def __init__(
        self,
        *args,
        batch_size: int,
        max_epochs: int = -1,
        distributed: Literal["dp"] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the dataloader.

        Args:
            batch_size: Batch size to use for training.
            max_epochs: Maximum number of epochs to train. **Specify this parameter only
                to the train data loader.**
            distributed: Distributed strategy to use for training. If training with single GPU,
                this value should be `None`; if training using data parallel with multi-GPU on
                a single node, this value should be `"dp"`.

        Raises:
            ValueError: `max_epochs` is specified when initializing the evaluation dataloader.
            RuntimeError: `torch.distributed` package is not available.
            RuntimeError: The default process group is not initialized. Make sure to call
                `torch.distributed.init_process_group` to initialize the default process
                group before doing a multiprocessing distributed training.
            ValueError: `self.sampler` is not an instance of `DistributedSampler`. An instance of
                `DistributedSampler` will shuffle and distribute data among GPUs, so it is required
                for data parallel training.
            ValueError: `DistributedSampler` passed in `self.sampler` is inconsistent with the default
                process group. Currently, we assume that all the GPUs in the node will be used for
                training. In this case, the instance of `DistributedSampler` should have
                `sampler.num_replicas == torch.distributed.get_world_size()`
                and `sampler.rank == torch.distributed.get_rank()`.
            TypeError: Parameter `distributed` is not correctly specified. Currently, it can only
                be set as `"dp"` or `None`.
            RuntimeError: Scaling is triggered when the profile window exceeds the number of iterations
                in one epoch. But latter is too small, so scaling can not produce a valid profile window.
                Please consider increasing batch size.
        """
        # Save attributes.
        self.batch_size = batch_size
        self.split = "train" if max_epochs != -1 else "eval"
        self.max_epochs = max_epochs
        self.log_prefix = f"[ZeusDataLoader({self.split})]"

        # Initialize the DataLoader.
        super().__init__(*args, batch_size=batch_size, **kwargs)

        # World size and rank for distributed training.
        # Set default value for single-GPU.
        self.world_size = 1
        self.rank = 0
        # Check whether we are doing a distributed training.
        # Pass in world size and rank.

        self.distributed = distributed

        if self.distributed == "dp":
            # Check if the distributed package is available.
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available.")
            # Check if the process group is initialized.
            if not dist.is_initialized():
                raise RuntimeError(
                    "Default process group has not been initialized,"
                    " please make sure to call `init_process_group`"
                    " before you instantiate `ZeusDataLoader`."
                )
            # Check if `self.sampler` is an instance of DistributedSampler.
            if not isinstance(getattr(self, "sampler", None), DistributedSampler):
                raise ValueError(
                    "Sampler is not an instance of `DistributedSampler`."
                    " Data parallel training on multi-GPU requires a `DistributedSampler`."
                )
            # Check the consistency between the sampler and process group.
            if (
                self.sampler.num_replicas != dist.get_world_size()
                or self.sampler.rank != dist.get_rank()
            ):
                raise ValueError(
                    "`DistributedSampler` is inconsistent with the default process group."
                    f" The default process group has `world_size={dist.get_world_size()}`,"
                    f" `rank={dist.get_rank()}`."
                )
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            if self.distributed is not None:
                raise TypeError(
                    '`distributed` currently only accepts `"dp"` or `None`.'
                )

        if self._is_train:
            self._log(
                f"Distributed data parallel: {'ON' if self.world_size > 1 else 'OFF'}"
            )

        if self._is_train:
            if ZeusDataLoader.train_batch_size != 0:
                # If max_epochs is specified when initializing a eval dataloader,
                # it will mistaken itself as a train dataloader.
                # In this case, raise a ValueError.
                raise ValueError("Specify max_epochs only to the train dataloader.")
            # In data parallel training, each DataLoader gets `batch_size=global_batch_size/num_gpus`.
            # So, we scale the `train_batch_size` for the consistency with ZeusMaster.
            # NOTE: Zeus assume `global_batch_size == batch_size * num_gpus`. So please ensure that
            # `global_batch_size` is divisible by `num_gpu` in the training script.
            ZeusDataLoader.train_batch_size = self.batch_size * self.world_size

        # Retrieve environment variables from ZeusMaster.
        self.target_metric = get_env("ZEUS_TARGET_METRIC", float)
        self.logdir = get_env("ZEUS_LOG_DIR", str, default="zeus_log")
        self.job_id = get_env("ZEUS_JOB_ID", str, default="zeus")
        self.cost_thresh = get_env("ZEUS_COST_THRESH", float, default=float("inf"))
        self.eta_knob = get_env("ZEUS_ETA_KNOB", float, default=0.5)
        self.monitor_path = get_env(
            "ZEUS_MONITOR_PATH",
            str,
            default="/workspace/zeus/zeus_monitor/zeus_monitor",
        )
        self.warmup_iter, self.profile_iter = map(
            int, get_env("ZEUS_PROFILE_PARAMS", str, default="10,40").split(",")
        )
        self.use_optimal_pl = get_env("ZEUS_USE_OPTIMAL_PL", bool, default=True)

        # Create ZEUS_LOG_DIR if it does not exist.
        os.makedirs(self.logdir, exist_ok=True)

        # Check whether the monitor path is good.
        if not os.access(self.monitor_path, os.X_OK):
            raise RuntimeError(f"'{self.monitor_path}' is not executable")

        # Whether the target metric was reached.
        self.target_metric_reached = False

        # Construct relevant paths.
        self.train_json = (
            f"{self.logdir}/{self.job_id}+bs{self.train_batch_size}.train.json"
        )
        self.power_json = f"{self.logdir}/bs{self.train_batch_size}.power.json"

        # Numbers related to the dataloader.
        # sample_num: the number of iterations processed in the current epoch.
        # num_samples: the total number of iterations in one epoch.
        self.epoch_num = 0
        self.sample_num = 0
        self.num_samples = len(self)

        # Pass the length of the eval dataloader for `epochs`.
        if not self._is_train:
            ZeusDataLoader.eval_num_samples = self.num_samples

        # If the number of iterations in one epoch (`num_samples`) is smaller than or equal
        # to one profile window (`warmup_iters + profile_iters`), we will not be able to
        # profile for any power limit. So, we scale the profile window to fit in one epoch.
        # We also avoid using the last batch of one epoch, becasue when `drop_last == True`,
        # the last batch will be smaller. This usually happens with large batch size on
        # small datasets, eg. CIFAR100.
        if self._is_train and self.warmup_iter + self.profile_iter >= self.num_samples:
            self._log(
                f"The profile window takes {self.warmup_iter + self.profile_iter}"
                f" iterations ({self.warmup_iter} for warmup + {self.profile_iter}"
                f" for profile) and exceeds the number of iterations ({self.num_samples})"
                f" in one epoch. Scaling the profile window to fit in one epoch..."
            )
            scaling_factor = (self.num_samples - 1) / (
                self.warmup_iter + self.profile_iter
            )
            self.warmup_iter = int(self.warmup_iter * scaling_factor)
            self.profile_iter = int(self.profile_iter * scaling_factor)
            if self.warmup_iter == 0 or self.profile_iter == 0:
                raise RuntimeError(
                    f"Number of iterations in one epoch is {self.num_samples} and"
                    " is too small for applying the scaling. Please consider using"
                    " a smaller batch size. If you are running `run_zeus.py`, please"
                    " pass a smaller value to `--b_max`."
                )
            self._log(
                f"Scaling done! New profile window takes {self.warmup_iter + self.profile_iter}"
                f" iterations ({self.warmup_iter} for warmup + {self.profile_iter} for profile)."
            )

        # Power profiling windows
        #
        # +----- warmup_start (change power limit)
        # |        +----- prof_start (record timestamp)
        # |        |                      +----- prof_end (compute and save results)
        # | warmup |        profile       |
        # v  iter  v          iter        v
        # ================================= =====================
        # |      power limit = 250W       | | power limit = 225W  ...
        # ================================= =====================
        #
        # =======================================================
        # |                         Epoch 1                       ...
        # =======================================================
        #
        # Initialize variables for profiling
        self.warmup_start_sample = 0
        self.prof_start_time = 0.0
        self.prof_start_sample = 0
        self.prof_state = NOT_PROFILING
        self.prof_pl_index = 0

        # Initialize data structure for storing the energy accounting
        # based on the number of GPUs.
        if self._is_train:
            # Sanity check
            assert self.world_size > 0, f"{self.world_size=}"
            assert self.max_epochs > 0, f"{self.max_epochs=}"
            ZeusDataLoader.train_epoch_energy = np.zeros(
                shape=(self.world_size, self.max_epochs), dtype=np.float64
            )
            ZeusDataLoader.eval_epoch_energy = np.zeros(
                shape=(self.world_size, self.max_epochs), dtype=np.float64
            )

        # Initialize NVML and get GPU handle or each GPU at the master process.
        self.gpu_handles = []
        pynvml.nvmlInit()
        for index in range(self.world_size):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            # Set persistent mode.
            pynvml.nvmlDeviceSetPersistenceMode(handle, pynvml.NVML_FEATURE_ENABLED)
            self.gpu_handles.append(handle)
        # Query NVML for the possible power limit range. Unit is mW.
        min_pl, self.max_pl = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(
            self.gpu_handles[0]
        )
        self.power_limits = list(range(self.max_pl, min_pl - 25_000, -25_000))
        if self._is_train:
            self._log(f"Power limit range: {self.power_limits}")

        # Check whether profiling is ON or OFF. If OFF, load the power limit
        # from power_json, and set power limit for all GPUs at the master process.
        if self._is_train and self.rank == 0:
            should_profile = self._should_profile
            self._log(f"Power profiling: {'ON' if should_profile else 'OFF'}")
            # If we need to do profiling, no need to touch the power limit.
            # If profiling is already done, load profile information from power_json.
            # Only then do we have the optimal PL available.
            # We only load in the train dataloader since it populates classvars.
            if not should_profile:
                self._load_power_results()
                self._set_gpu_steady_power_limit()

        # Make sure NVML is shutdown and the monitor is killed when the training script exits.
        if self._is_train:

            def exit_hook():
                pynvml.nvmlShutdown()
                # Master process kills all the monitors.
                if self.rank == 0:
                    for index, monitor in enumerate(self.monitors):
                        monitor.kill()
                        self._log(f"[GPU_{index}] Stopped Zeus monitor.")

            atexit.register(exit_hook)

    def epochs(self) -> Generator[int, None, None]:
        """Yield the current epoch number from 0 until when training should stop.

        Training should stop when

        - the cost reached the cost threshold, or
        - the maximum number of epochs was reached, or
        - the target metric was reached.

        When done, stores the job results in `train_json`.

        Yields:
            Epoch indices starting from zero.

        Raises:
            ZeusCostThresholdExceededException: the predicted cost after the next
                epoch exceeds the cost threshold. When doing data parallel training,
                this exception is used for ternimating all the processes.
        """
        # Sanity check.
        if not self._is_train:
            raise RuntimeError("Use epochs() on the train dataloader.")

        while True:
            # Variables for storing time/energy consumption & cost
            time_consumed, energy_consumed = -1, -1
            cost = -1
            if self.rank == 0:
                # Sanity checks.
                enum = self.epoch_num
                assert (
                    len(self.train_epoch_time) == enum
                ), f"{len(self.train_epoch_time)=}"
                assert (
                    len(self.eval_epoch_time) == enum
                ), f"{len(self.eval_epoch_time)=}"

                # Compute time and energy consumption up to now.
                # Compute time consumption at GPU_0
                time_consumed = sum(self.train_epoch_time + self.eval_epoch_time)
                # Compute energy consumption over all the GPUs
                energy_consumed = (
                    self.train_epoch_energy.sum() + self.eval_epoch_energy.sum()
                )
                cost = zeus_cost(
                    energy_consumed,
                    time_consumed,
                    self.eta_knob,
                    self.max_pl // 1000 * self.world_size,
                )
                self._log(
                    f"Up to epoch {self.epoch_num}: "
                    f"time={time_consumed:.2f}, energy={energy_consumed:.2f}, cost={cost:.2f}"
                )

            # target_metric_reached is set when the current validation metric is reported to
            # the train dataloader after the end of each epoch.
            # Stop if the target metric was reached.
            if self.target_metric_reached:
                if self.rank == 0:
                    # Sanity check that time/energy consumption & cost are valid in master process.
                    assert time_consumed >= 0 and energy_consumed >= 0 and cost >= 0
                    self._log(
                        f"Target metric {self.target_metric} was reached! Stopping."
                    )
                    self._save_train_results(energy_consumed, time_consumed, cost, True)
                return

            # Max epoch is a hard stop.
            if self.epoch_num >= self.max_epochs:
                if self.rank == 0:
                    # Sanity check that time/energy consumption & cost are valid in master process.
                    assert time_consumed >= 0 and energy_consumed >= 0 and cost >= 0
                    self._log(
                        f"Maximum number of epochs {self.max_epochs} reached. Stopping."
                    )
                    self._save_train_results(
                        energy_consumed, time_consumed, cost, False
                    )
                return

            # No need to do anything in the first epoch.
            if self.epoch_num == 0:
                yield 0
                continue

            # Just continue if we're profiling.
            # This will ignore and continue training even if the cost threshold was exceeded.
            # However, the profiling cost actually exceeding the cost threshold would not
            # happen frequently. It's more like a wrong cost threshold.
            if self._should_profile:
                if cost >= self.cost_thresh:
                    self._log(
                        f"{cost=:.2f} exceeded threshold {self.cost_thresh:.2f} at GPU_{self.rank}, "
                        "but just continue since we're profiling."
                    )
                yield self.epoch_num
                continue

            if self.rank == 0:
                # Sanity check that time/energy consumption & cost are valid in master process.
                assert time_consumed >= 0 and energy_consumed >= 0 and cost >= 0

                # We want to predict whether running the next epoch will exceed the cost threshold.
                next_train_time = (
                    self.num_samples / self.train_tput_result[self.optimal_pl]
                )
                next_eval_time = (
                    self.eval_num_samples / self.eval_tput_result[self.optimal_pl]
                )
                next_time = next_train_time + next_eval_time
                next_train_energy = (
                    next_train_time * self.train_power_result[self.optimal_pl]
                )
                next_eval_energy = (
                    next_eval_time * self.eval_power_result[self.optimal_pl]
                )
                next_energy = next_train_energy + next_eval_energy
                self._log(
                    f"Optimal PL train & eval expected time={next_time:.2f} energy={next_energy:.2f}"
                )
                next_time_consumed = time_consumed + next_time
                next_energy_consumed = energy_consumed + next_energy
                next_cost = zeus_cost(
                    next_energy_consumed,
                    next_time_consumed,
                    self.eta_knob,
                    self.max_pl // 1000 * self.world_size,
                )
                self._log(
                    f"Expected next epoch: time={next_time_consumed:.2f}, "
                    f"energy={next_energy_consumed:.2f}, "
                    f"cost={next_cost:.2f}"
                )

                # Stop if the predicted cost of the next epoch exceeds the cost threshold.
                if next_cost >= self.cost_thresh:
                    # Save training results
                    self._save_train_results(
                        energy_consumed, time_consumed, cost, False
                    )
                    # NOTE: We use a customized exception to terminate ALL the processes for
                    # the purpose of multiprocessing management.
                    # When doing data parallel training on multiple processes, ONLY the master
                    # process will predict `next_cost` and do the threshold checking. However,
                    # once the predicted cost exceeds the threshold, we want to terminate ALL
                    # the processes. Currently this is achieved by throwing an exception at the
                    # master process. The lauching script will terminate all the processes that
                    # are still alive.
                    raise ZeusCostThresholdExceededException(
                        time_consumed,
                        energy_consumed,
                        cost,
                        next_cost,
                        self.cost_thresh,
                    )

            yield self.epoch_num

    def report_metric(self, metric: float, higher_is_better: bool) -> None:
        """Report the validation metric to the train dataloader.

        If doing data parallel training, please make sure
        to call `dist.all_reduce()` to reduce the validation metric across all GPUs
        before calling `train_loader.report_metric()`.

        Args:
            metric: The validation metric of the current epoch.
            higher_is_better: For example, this should be `True` for accuracy
                and `False` for error.
        """
        assert self._is_train, "Use report_metric on the train dataloader."
        if higher_is_better:
            if metric >= self.target_metric:
                self.target_metric_reached = True
        else:
            if metric <= self.target_metric:
                self.target_metric_reached = True

    @property
    def _should_profile(self) -> bool:
        """Whether profiling is not done."""
        return not Path(self.power_json).exists()

    @property
    def _power_limits_left(self) -> bool:
        """Whether there are power limits left to profile."""
        return self.prof_pl_index < len(self.power_limits)

    def _compute_optimal_pl(self) -> int:
        """Return the cost-optimal power limit."""
        # Sanity checks.
        assert ZeusDataLoader.train_tput_result
        assert ZeusDataLoader.train_power_result
        # Only compute optimal PL at master process.
        assert self.rank == 0

        # Compute power cost
        tput = ZeusDataLoader.train_tput_result
        power = ZeusDataLoader.train_power_result
        cost_map = {
            pl: (
                self.eta_knob * power[pl]
                + (1 - self.eta_knob) * self.max_pl * self.world_size
            )
            / tput[pl]
            for pl in self.power_limits
        }
        optimal_pl = min(cost_map.keys(), key=cost_map.get)  # type: ignore
        self._log(f"Cost-optimal power limit is {optimal_pl//1000}W")
        return optimal_pl

    def _set_gpu_power_limit(self, power_limit: int) -> None:
        """Set the GPU's power limit using NVML.

        This method only invokes NVML when `power_limit` is not the same as
        the current GPU power limit.

        Args:
            power_limit: Power limit to set.
        """
        # Sanity check.
        # Only set power limit at master process.
        assert self.rank == 0
        assert len(self.gpu_handles) == self.world_size

        # Set power limit for all GPUs.
        if self.current_gpu_pl != power_limit:
            for index in range(self.world_size):
                pynvml.nvmlDeviceSetPowerManagementLimit(
                    self.gpu_handles[index], power_limit
                )
                self._log(f"[GPU_{index}] Set GPU power limit to {power_limit//1000}W.")
            ZeusDataLoader.current_gpu_pl = power_limit

    def _set_gpu_steady_power_limit(self) -> None:
        """Set the steady power limit based on self.use_optimal_pl."""
        # Sanity check.
        # Only set power limit at master process.
        assert self.rank == 0

        power_limit = ZeusDataLoader.optimal_pl if self.use_optimal_pl else self.max_pl
        self._log(
            "Steady state power limit: "
            f"{'OPT' if self.use_optimal_pl else 'MAX'} {power_limit//1000}W"
        )
        self._set_gpu_power_limit(power_limit)

    def _log(
        self, message: str, level: int = logging.INFO, master_only: bool = True
    ) -> None:
        """Print out message with prefix.

        Args:
            message: The message to log out.
            level: The logging level to use. (Default: `logging.INFO`)
            master_only: Whether only logged by master process. Usually set to True for the
                global logging and False for the GPU-specific logging . If set to False,
                a prefix indicates which GPU this log comes from will be included as well.
                (Default: `True`)
        """
        if master_only:
            if self.rank == 0:
                LOG.log(level, "%s %s", self.log_prefix, message)
        else:
            gpu_log_prefix = f"[GPU_{self.rank}]"
            LOG.log(level, "%s %s %s", self.log_prefix, gpu_log_prefix, message)

    @cached_property
    def _is_train(self) -> bool:
        """Return whether this dataloader is for training."""
        return self.split == "train"

    def _power_log_path(self, rank: int) -> str:
        """Build the path for the power monitor log file at the GPU with rank."""
        return f"{self.logdir}/bs{self.train_batch_size}+e{self.epoch_num}+gpu{rank}.power.log"

    def _start_monitor(self) -> None:
        """Start the power monitor subprocess."""
        # Sanity checks.
        assert not ZeusDataLoader.monitors
        # Only the master process starts the monitors.
        assert self.rank == 0

        # Start monitors. Master process starts and records all monitors.
        for index in range(self.world_size):
            monitor = subprocess.Popen(
                [self.monitor_path, self._power_log_path(index), "0", "100"],
            )
            self._log(f"[GPU_{index}] Zeus monitor started.")
            ZeusDataLoader.monitors.append(monitor)

    def _kill_monitor(self) -> None:
        """Kill the power monitor subprocess."""
        # Sanity checks.
        assert ZeusDataLoader.monitors
        # Only the master process kills the monitors.
        assert self.rank == 0

        # Kill monitors.
        for monitor in ZeusDataLoader.monitors:
            monitor.send_signal(signal.SIGINT)
        for monitor in ZeusDataLoader.monitors:
            monitor.wait(timeout=1.0)

        # Cleanup the monitor list
        ZeusDataLoader.monitors = []

    def _start_warmup(self) -> None:
        """Let the GPU run for some time with the poewr limit to profile."""
        # Sanity checks.
        assert self._should_profile, f"start_warmup: {self._should_profile=}"
        assert self._is_train, f"start_warmup: {self._is_train=}"
        assert self._power_limits_left, f"start_warmup: {self._power_limits_left=}"
        # Sanity check that this profile window ends before the end of the current epoch.
        assert (
            self.sample_num + self.warmup_iter + self.profile_iter < self.num_samples
        ), (
            "start_warmup: "
            f"end_of_this_profile_window {self.sample_num + self.warmup_iter + self.profile_iter} "
            f"< end_of_this_epoch {self.num_samples}"
        )

        # Call cudaSynchronize to make sure this is the iteration boundary.
        torch.cuda.synchronize()

        # Change power limit.
        if self.rank == 0:
            power_limit = self.power_limits[self.prof_pl_index]
            self._set_gpu_power_limit(power_limit)

            self._log(f"Warm-up started with power limit {self.current_gpu_pl//1000}W")

        self.warmup_start_sample = self.sample_num

        # Set profiling state.
        self.prof_state = WARMING_UP

    def _start_prof(self) -> None:
        """Start profiling power consumption for the current power limit."""
        # Sanity checks.
        assert self._should_profile, f"start_prof: {self._should_profile=}"
        assert self._is_train, f"start_prof: {self._is_train=}"
        assert self._power_limits_left, f"start_prof: {self._power_limits_left=}"
        # Sanity check that this profile window ends before the end of the current epoch.
        assert self.sample_num + self.profile_iter < self.num_samples, (
            "start_prof: "
            f"end_of_this_profile_window {self.sample_num + self.profile_iter} "
            f"< end_of_this_epoch {self.num_samples}"
        )

        # Start profile timer.
        self.prof_start_time = time.monotonic()

        # Set the sample number when we started profiling.
        self.prof_start_sample = self.sample_num

        # Set profiling state.
        self.prof_state = PROFILING

        self._log(f"Profile started with power limit {self.current_gpu_pl//1000}W")

    def _end_prof(self) -> None:
        """End profiling power consumption for this power limit.

        Raises:
            ValueError: ValueError raised by sklearn.metrics.auc in analyze.avg_power,
                might due to profile window too small. In this case, user should consider
                increasing profile window.
        """
        # Sanity checks.
        assert self._should_profile, f"end_prof: {self._should_profile=}"
        assert self._is_train, f"end_prof: {self._is_train=}"
        assert self._power_limits_left, f"end_prof: {self._power_limits_left=}"
        # Sanity check that this profile window ends before the end of the current epoch.
        assert self.sample_num < self.num_samples, (
            "end_prof: "
            f"end_of_this_profile_window {self.sample_num} "
            f"< end_of_this_epoch {self.num_samples}"
        )

        # Set profiling state.
        self.prof_state = NOT_PROFILING

        # Call cudaSynchronize to make sure this is the iteration boundary.
        torch.cuda.synchronize()

        # Freeze time.
        now = time.monotonic()

        # Advance to the next power limit. Affects self.power_limits_left.
        self.prof_pl_index += 1

        if self.rank == 0:
            # Summing up the average power on all GPUs.
            sum_avg_power = 0
            for index in range(self.world_size):
                # Compute and save average power.
                # The monitor is still running, so we just integrate from the beginning
                # of this profiling window (of course exclude warmup) up to now.
                # The power log file only records for the current epoch,
                # so we compute an offset.
                try:
                    avg_power = analyze.avg_power(
                        self._power_log_path(index),
                        start=self.prof_start_time - self.epoch_start_time,
                    )
                except ValueError:
                    self._log(
                        "ValueError from analyze.avg_power, please consider increasing self.profile_iter.",
                        logging.ERROR,
                    )
                    raise
                sum_avg_power += avg_power
            self.train_power_result[self.current_gpu_pl] = sum_avg_power

            # Compute and save throughput. We use the time at the master process.
            time_consumed = now - self.prof_start_time
            samples_processed = self.sample_num - self.prof_start_sample
            throughput = samples_processed / time_consumed
            self.train_tput_result[self.current_gpu_pl] = throughput

            self._log(f"Profile done with power limit {self.current_gpu_pl//1000}W")

            # If we're done with all power limits, compute the optimal power limit
            # and change to that power limit for the rest of the epoch.
            # This will lead to the eval epoch being run with the optimal power limit,
            # and since self.should_profile is still True, tput/power will be profiled.
            # Profiling the optimal power limit on eval set will help us better predict
            # the time and energy consumed in the next eval epoch, to help us decide
            # whether running next epoch will exceed the cost threshold.
            if not self._power_limits_left:
                self._log("This was the last power limit to explore.")
                ZeusDataLoader.optimal_pl = self._compute_optimal_pl()
                self._set_gpu_power_limit(ZeusDataLoader.optimal_pl)

    def _save_power_results(self) -> None:
        """Write the power profiling results to `power_json`."""
        # Sanity check.
        # Only save power results at master process.
        assert self.rank == 0

        prof_result = dict(
            job_id=self.job_id,  # Not used. Just for the purpose of record.
            train_power=self.train_power_result,
            train_throughput=self.train_tput_result,
            eval_power=self.eval_power_result,
            eval_throughput=self.eval_tput_result,
            optimal_pl=self.optimal_pl,
        )
        # NOTE: Write-then-move needed if we're handling concurrent jobs.
        with open(self.power_json, "w") as f:
            json.dump(prof_result, f)
        with open(self.power_json, "r") as f:
            self._log("Power profiling done.")
            self._log(f"Saved {self.power_json}: {f.read()}")

    def _load_power_results(self) -> None:
        """Load power profiling information into the class from `power_json`."""
        # Sanity check.
        # Only load power results at master process.
        assert self.rank == 0

        # Helper function that casts the keys of a dictionary to integer.
        def as_int_key(dictionary: dict[str, float]) -> dict[int, float]:
            result = {}
            for key, value in dictionary.items():
                result[int(key)] = value
            return result

        with open(self.power_json, "r") as f:
            power_results = json.load(f)

        ZeusDataLoader.train_power_result = as_int_key(power_results["train_power"])
        ZeusDataLoader.train_tput_result = as_int_key(power_results["train_throughput"])
        ZeusDataLoader.eval_power_result = as_int_key(power_results["eval_power"])
        ZeusDataLoader.eval_tput_result = as_int_key(power_results["eval_throughput"])
        ZeusDataLoader.optimal_pl = power_results["optimal_pl"]

        self._log(f"Loaded {self.power_json}: {power_results}")

    def _save_train_results(
        self, energy: float, time_: float, cost: float, reached: bool
    ) -> None:
        """Write the job training results to `train_json`."""
        # Sanity check.
        # Only load power results at master process.
        assert self.rank == 0

        train_result = dict(
            energy=energy,
            time=time_,
            cost=cost,  # Not used. Just for reference.
            num_epochs=self.epoch_num,  # Not used. Just for reference.
            reached=reached,
        )
        with open(self.train_json, "w") as f:
            json.dump(train_result, f)
        with open(self.train_json, "r") as f:
            self._log("Training done.")
            self._log(f"Saved {self.train_json}: {f.read()}")

    # pylint: disable=attribute-defined-outside-init
    def __iter__(self):
        """Signal the beginning of an epoch."""
        # Sanity check that there is no incomplete profile window at the beginning of epoch,
        # because we start profiling only if the entire profiling window can fit in the rest of
        # the training epoch.
        assert self.prof_state == NOT_PROFILING, f"__iter__: {self.prof_state=}"

        # Update counters.
        self.epoch_num += 1
        self.sample_num = 0
        self._log(f"Epoch {self.epoch_num} begin.")

        # Start epoch timer.
        self.epoch_start_time = time.monotonic()

        # Cache the dataloader iterator.
        self.iter = super().__iter__()

        # The power limit of the GPU is only changed by the train dataloader.
        if self._is_train and self.rank == 0:
            # The train loader always starts the monitor, and the eval loader kills it.
            self._start_monitor()
            # If we're not profiling, use the steady state power limit.
            # If we are profiling, the power limit will be set in __next__ with warmup.
            # Power limit result is already loaded in when initializing the train dataloader,
            # so we just set the power limit directly.
            if not self._should_profile:
                self._set_gpu_steady_power_limit()

        return self

    def __next__(self):
        """Signal the beginning of an iteration."""
        # Update counters.
        self.sample_num += 1

        # Try to fetch next batch.
        try:
            data = next(self.iter)
        except StopIteration:
            # End of this epoch.
            # Sanity check that there is no incomplete profile window at the end of epoch.
            assert self.prof_state == NOT_PROFILING, f"__next__: {self.prof_state=}"

            # Make sure all GPU operations are done so that now is the *actual* end of this epoch.
            torch.cuda.synchronize()

            # The eval dataloader kills the monitor.
            if not self._is_train and self.rank == 0:
                self._kill_monitor()

            # Compute epoch time and energy consumption.
            # We're interested in the actual time/energy consumption here.
            #
            #   <----------------- monitor lifetime ------------------>
            #
            #   =======================================================
            #   |                Train                    ||   Eval   |
            #   =======================================================
            #   ^                                         ^^          ^
            #   |                                        / |          |
            #   epoch_start_time          time.monotonic() |          |
            #   for train loader          for train loader |          |
            #                                              |          |
            #                               epoch_start_time   time.monotonic()
            #                                for eval loader   for eval loader
            #
            if self.rank == 0:
                # Sanity check that `epoch_num` is within valid range
                assert self.epoch_num >= 1, f"__next__: {self.epoch_num=}"
                # Compute the time/energy consumption for this epoch.
                time_consumption = time.monotonic() - self.epoch_start_time
                if self._is_train:
                    self.train_epoch_time.append(time_consumption)
                    # Record the energy consumption for each GPU.
                    for index in range(self.world_size):
                        # The monitor is still running, and we integrate over the entire log.
                        energy_consumption = analyze.energy(self._power_log_path(index))
                        self.train_epoch_energy[index][
                            self.epoch_num - 1
                        ] = energy_consumption
                else:
                    # We just killed the monitor. Integrate the last time_consumption seconds.
                    self.eval_epoch_time.append(time_consumption)
                    sum_energy_consumption = 0
                    # Record the energy consumption for each GPU.
                    for index in range(self.world_size):
                        # We set the `start` to exclude the logs during train time.
                        energy_consumption = analyze.energy(
                            self._power_log_path(index), start=-time_consumption
                        )
                        self.eval_epoch_energy[index][
                            self.epoch_num - 1
                        ] = energy_consumption
                        sum_energy_consumption += energy_consumption
                    # For the eval dataloader, we want to record the throughput and power
                    # for the current power limit. Since the train dataloader sets the power limit
                    # to the optimal power limit right after profiling is done, this will naturally
                    # record the tput/power of the optimal PL. From the following epochs where we
                    # don't profile anything, we directly use these values to compute the time and
                    # energy consumed.
                    if self._should_profile:
                        self.eval_tput_result[self.current_gpu_pl] = (
                            self.num_samples / time_consumption
                        )
                        self.eval_power_result[self.current_gpu_pl] = (
                            sum_energy_consumption / time_consumption
                        )
                        # The optimal PL being known means that all power limits have been explored.
                        # Let us end profiling by writing profile information to `power_json`.
                        if self.optimal_pl != 0:
                            self._save_power_results()
                self._log(
                    f"{self.split} epoch {self.epoch_num} done: "
                    f"time={time_consumption:.2f} energy={energy_consumption:.2f}"
                )

            # Re-raise StopIteration.
            raise

        # We're in the middle of an epoch. The train loader has power limits left to profile.
        if self._is_train and self._should_profile and self._power_limits_left:
            # We weren't doing anything. Start warming up if the iterations left in
            # the current epoch can accommodate at least one profile window.
            if (
                self.prof_state == NOT_PROFILING
                and self.sample_num + self.warmup_iter + self.profile_iter
                < self.num_samples
            ):
                self._start_warmup()
            # We're done warming up. Start the actual profiling window.
            elif (
                self.prof_state == WARMING_UP
                and self.sample_num - self.warmup_start_sample == self.warmup_iter
            ):
                self._start_prof()
            # We're done profiling. Stop the profiling window and gather results.
            elif (
                self.prof_state == PROFILING
                and self.sample_num - self.prof_start_sample == self.profile_iter
            ):
                self._end_prof()

        return data
