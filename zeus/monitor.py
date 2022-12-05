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

"""Helpers for using the Zeus monitor inside training scripts for energy profiling."""

from __future__ import annotations

import time
import atexit
import signal
import tempfile
import subprocess
from typing import Generator
from contextlib import contextmanager

import torch

from zeus.analyze import energy as compute_energy


class ZeusMonitorContext:
    """Monitors the energy and time consumption inside a training loop.

    Skip the first `skip_steps` steps and profile for the next `profile_steps`. Does
    nothing after that, before you call [`reset`][zeus.monitor.ZeusMonitorContext.reset].

    You can check whether profiling is done (i.e., `skip_steps + profile_steps` passed)
    through [`is_done`][zeus.monitor.ZeusMonitorContext.is_done] and query results through
    [`total_energy`][zeus.monitor.ZeusMonitorContext.total_energy],
    [`avg_energy`][zeus.monitor.ZeusMonitorContext.avg_energy],
    [`total_time`][zeus.monitor.ZeusMonitorContext.total_time], and
    [`avg_time`][zeus.monitor.ZeusMonitorContext.avg_time].

    ## Integration example

    ```python
    zeus_ctx = zeus.monitor.ZeusMonitorContext(skip_steps=10, profile_steps=10)

    for step, (x, y) in enumerate(train_dataloader):
        print(f"Training step {step}.")

        # Wrap the code range of one training step.
        # zeus_ctx will ignore the first 10 steps and measure the next 10 steps.
        with zeus_ctx.step():
            training_step(x, y)

        # An alternative way if you don't want to use context managers.
        zeus_ctx.start_step()
        training_step(x, y)
        zeus_ctx.finish_step()

        # Check if 20 steps passed and query results.
        if zeus_ctx.is_done:
            print(
                f"{zeus_ctx.profile_steps} training steps "
                f"consumed {zeus_ctx.total_energy} Joules in {zeus_ctx.total_time} seconds."
            )
            zeus_ctx.reset()
    ```
    """

    def __init__(
        self,
        skip_steps: int = 10,
        profile_steps: int = 10,
        device_id: int = 0,
        zeus_monitor_path: str = "zeus_monitor",
        zeus_monitor_sleep_ms: int = 0,
        zeus_monitor_log_dir: str | None = None,
    ) -> None:
        """Create a Zeus monitor context.

        Args:
            skip_steps: The number of steps to skip when profiling energy consumption.
            profile_steps: The number of steps to profile and average over for energy consumption.
            device_id: CUDA device ID to run the monitor for.
            zeus_monitor_path: `argv[0]` to use when spawning the Zeus monitor.
            zeus_monitor_sleep_ms: How long the Zeus monitor should sleep after sampling power.
            zeus_monitor_log_dir: The directory to put the monitor log file. A temporary file is
                used if not specified.
        """
        # Save arguments.
        self.skip_steps = skip_steps
        self.profile_steps = profile_steps
        self.device_id = device_id

        # Spawn the Zeus monitor.
        if zeus_monitor_log_dir:
            self._power_csv = f"{zeus_monitor_log_dir}/gpu{self.device_id}.power.csv"
        else:
            self._power_csv = tempfile.mkstemp(
                suffix=f"+gpu{self.device_id}.power.csv"
            )[1]
        self._monitor = subprocess.Popen(
            args=[
                zeus_monitor_path,
                self._power_csv,
                "0",
                str(zeus_monitor_sleep_ms),
                str(self.device_id),
            ],
            stdin=subprocess.DEVNULL,
        )
        self._time_origin = time.monotonic()

        # Make sure the monitor is eventually stopped.
        def exit_hook():
            self._monitor.send_signal(signal.SIGINT)
            time.sleep(2.0)
            self._monitor.kill()

        atexit.register(exit_hook)

        # Set internal profiling states.
        self._started_steps: int = 0
        self._finished_steps: int = 0
        self._profile_start_times: list[float] = []
        self._profile_end_times: list[float] = []
        self._metric_cache: dict[str, float] = {}

    def start_step(self) -> None:
        """Mark the beginning of one step."""
        self._started_steps += 1
        if (
            self.skip_steps
            < self._started_steps
            <= self.skip_steps + self.profile_steps
        ):
            torch.cuda.synchronize()
            current_time = time.monotonic()
            self._profile_start_times.append(current_time - self._time_origin)

    def finish_step(self) -> None:
        """Mark the end of one step."""
        self._finished_steps += 1
        if (
            self.skip_steps
            < self._finished_steps
            <= self.skip_steps + self.profile_steps
        ):
            torch.cuda.synchronize()
            current_time = time.monotonic()
            self._profile_end_times.append(current_time - self._time_origin)

    @contextmanager
    def step(self) -> Generator[None, None, None]:
        """Wrap one training step to mark start and finish times."""
        try:
            self.start_step()
            yield
        finally:
            self.finish_step()

    @property
    def is_done(self) -> bool:
        """Return whether the specified profiling steps are done."""
        if self._started_steps != self._finished_steps:
            raise RuntimeError(
                "`is_done` should be called outside of the code range marked as a 'step'."
            )
        return self._finished_steps >= self.skip_steps + self.profile_steps

    @property
    def total_energy(self) -> float:
        """Return the total energy consumption of `profile_steps` steps."""
        if not self.is_done:
            raise RuntimeError("Metrics are accessible only after profiling is done.")
        try:
            return self._metric_cache["total_energy"]
        except KeyError:
            metric = sum(
                compute_energy(self._power_csv, start=start, end=end)
                for start, end in zip(
                    self._profile_start_times, self._profile_end_times
                )
            )
            self._metric_cache["total_energy"] = metric
            return metric

    @property
    def avg_energy(self) -> float:
        """Return the average energy consumption over `profiler_steps` steps."""
        if not self.is_done:
            raise RuntimeError("Metrics are accessible only after profiling is done.")
        try:
            return self._metric_cache["avg_energy"]
        except KeyError:
            metric = self.total_energy / self.profile_steps
            self._metric_cache["avg_energy"] = metric
            return metric

    @property
    def total_time(self) -> float:
        """Return the total time consumption of `profile_steps` steps."""
        if not self.is_done:
            raise RuntimeError("Metrics are accessible only after profiling is done.")
        try:
            return self._metric_cache["total_time"]
        except KeyError:
            metric = sum(
                end - start
                for start, end in zip(
                    self._profile_start_times, self._profile_end_times
                )
            )
            self._metric_cache["total_time"] = metric
            return metric

    @property
    def avg_time(self) -> float:
        """Return the average time consumption over `profiler_steps` steps."""
        if not self.is_done:
            raise RuntimeError("Metrics are accessible only after profiling is done.")
        try:
            return self._metric_cache["avg_time"]
        except KeyError:
            metric = self.total_time / self.profile_steps
            self._metric_cache["avg_time"] = metric
            return metric

    def reset(self) -> None:
        """Reset internal profile states."""
        self._started_steps = 0
        self._finished_steps = 0
        self._profile_start_times = []
        self._profile_end_times = []
        self._metric_cache = {}
