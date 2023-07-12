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

"""`PowerLimitOptimizer` finds a global GPU power limit that minimizes cost during training."""

from __future__ import annotations

import json
import atexit
from pathlib import Path
from dataclasses import dataclass

import pynvml
from zeus.callback import Callback

from zeus.monitor import ZeusMonitor
from zeus.util.logging import get_logger
from zeus.util.metric import zeus_cost


@dataclass
class Ready:
    """State for when we are ready to start measuring the next power limit.

    Initial state of the state machine if no previous profiling results were given.
    `Ready` -> `Warmup` on next `on_step_begin`.
    """

    next_power_limit: int


@dataclass
class Warmup:
    """State for when we are warming up for a power limit.

    `Warmup` -> `Profiling` on the `steps`'th `on_step_begin`.
    `Warmup` -> `Ready` on `on_epoch_end` before `steps`'th `on_step_end`.
    """

    current_power_limit: int
    steps: int


@dataclass
class Profiling:
    """State for when we are profiling a power limit.

    `Profiling` -> `Warmup` after `steps`'th `on_step_begin` and
        there are still power limits left to profile.
    `Profiling` -> `Done` after `steps`'th `on_step_begin` and
        there are no more power limits left to profile.
    `Profiling` -> `Ready` on `on_epoch_end` before `steps`'th `on_step_begin`.
    """

    current_power_limit: int
    steps: int


@dataclass
class Done:
    """State for when we are done profiling all power limits.

    Initial state of the state machine if previous profiling results were given.
    Final state of the state machine in any case.
    """

    optimal_power_limit: int


@dataclass
class Measurement:
    """POD for GPU energy and time measurements for one power limit."""

    power_limit: int  # In Watts.
    energy: float
    time: float


class GlobalPowerLimitOptimizer(Callback):
    """Optimizer for the power limit knob.

    This optimizer uses the JIT profiling log to determine the optimal power limit.
    """

    def __init__(
        self,
        monitor: ZeusMonitor,
        eta_knob: float = 0.5,
        warmup_steps: int = 10,
        profile_steps: int = 40,
        pl_step: int = 25,
        profile_path: str | Path | None = None,
    ) -> None:
        r"""Initialize the optimizer.

        GPU indices to profile and optimize for are taken from `monitor.gpu_indices`.

        Args:
            monitor: `ZeusMonitor` instance used to profile GPU time and energy consumption.
            eta_knob: The $0 \le \eta \le 1$ knob for the Zeus time-energy cost function.
            warmup_steps: Number of warmup iterations for each power limit.
            profile_steps: Number of profie iterations for each power limit.
            pl_step: The stride between power limits to explore, in unites of Watts.
            profile_path: If the path points to an existing file, load the profile from the file
                and do not run any profiling. If the path points to a non-existing file, profile
                and save the profile to the file. If `None`, do not save or load any profile.
        """
        self.monitor = monitor
        self.eta_knob = eta_knob
        self.warmup_steps = warmup_steps
        self.profile_steps = profile_steps
        self.pl_step = pl_step * 1000  # Internally, we use milliWatts.
        self.profile_path = (
            Path(profile_path) if isinstance(profile_path, str) else profile_path
        )

        # Sanity checks.
        if self.eta_knob < 0 or self.eta_knob > 1:
            raise ValueError("eta_knob must be between 0 and 1, inclusive both sides.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if self.profile_steps <= 0:
            raise ValueError("profile_steps must be positive.")
        if self.pl_step <= 0:
            raise ValueError("pl_step must be positive.")

        # Setup logging.
        self.logger = get_logger(type(self).__name__)

        # Set the range of power limits to explore.
        # Assert that supported power limits ranges are uniform across GPUs.
        pynvml.nvmlInit()
        pls = []
        self.handles = []
        for index in monitor.nvml_gpu_indices:
            device = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.handles.append(device)
            pls.append(pynvml.nvmlDeviceGetPowerManagementLimitConstraints(device))
        if not all(pls[0] == pl for pl in pls):
            raise ValueError("Power limits ranges are not uniform across GPUs.")
        self.power_limits = list(
            range(pls[0][1], pls[0][0] - self.pl_step, -self.pl_step)
        )

        # Turn on persistence mode and set to the highest power limit.
        try:
            for handle in self.handles:
                pynvml.nvmlDeviceSetPersistenceMode(handle, pynvml.NVML_FEATURE_ENABLED)
        except pynvml.NVMLError_NoPermission:  # type: ignore
            raise RuntimeError(
                "SYS_ADMIN capability is required to modify GPU power limits. "
                "Using --cap-add SYS_ADMIN when running the Docker container "
                "is the easiest way to do this."
            ) from None
        self.current_power_limit = 0

        # Store `Measurement` objects in a list, one for each power limit.
        self.measurements: list[Measurement] = []

        # State for the profiler state machine.
        self.state: Ready | Warmup | Profiling | Done

        # Initialize JIT profiling states.
        if self.profile_path is None:
            self.logger.info("JIT profiling enabled.")
            self.state = Ready(next_power_limit=self.power_limits[0])
            self.logger.info("Set power limit to the maximum before starting.")
            self._set_power_limit(max(self.power_limits))
        elif not self.profile_path.exists():
            self.logger.info(
                "JIT Profiling enabled. Profile will be saved to '%s'.",
                str(self.profile_path),
            )
            self.state = Ready(next_power_limit=self.power_limits[0])
            self.logger.info("Set power limit to the maximum before starting.")
            self._set_power_limit(max(self.power_limits))
        else:
            measurements = json.load(self.profile_path.open())["measurements"]
            self.measurements = [Measurement(**m) for m in measurements]
            self.logger.info(
                "Loaded previous profiling results from '%s'.", str(self.profile_path)
            )
            optimal_power_limit = self._compute_optimal_power_limit()
            self.logger.info(
                "Optimal power limit is %d W for eta_knob %f.",
                optimal_power_limit // 1000,
                self.eta_knob,
            )
            self.state = Done(optimal_power_limit=optimal_power_limit)
            self._set_power_limit(self.state.optimal_power_limit)

        # Restore all GPUs back to their maximum power limit on exit.
        atexit.register(lambda: self._set_power_limit(max(self.power_limits)))

    def on_epoch_end(self) -> None:
        """Mark the end of a training epoch."""
        if isinstance(self.state, Ready):
            pass

        elif isinstance(self.state, (Warmup, Profiling)):
            # Warmup/Profiling stage interrupted by the end of an epoch.
            self.logger.info(
                "%s phase for %d W interrupted by the end of a training epoch.",
                type(self.state).__name__,
                self.state.current_power_limit // 1000,
            )
            if isinstance(self.state, Profiling):
                self.monitor.end_window(
                    f"__PowerLimitOptimizer_{self.state.current_power_limit // 1000}",
                    cancel=True,
                )
            self.state = Ready(next_power_limit=self.state.current_power_limit)
            self._set_power_limit(max(self.power_limits))

        elif isinstance(self.state, Done):
            pass

    def on_step_begin(self) -> None:
        """Mark the beginning of a training step."""
        if isinstance(self.state, Ready):
            self.logger.info(
                "Starting warmup for power limit %d W.",
                self.state.next_power_limit // 1000,
            )
            self._set_power_limit(self.state.next_power_limit)
            self.state = Warmup(
                current_power_limit=self.state.next_power_limit,
                steps=self.warmup_steps,
            )

        elif isinstance(self.state, Warmup):
            self.state.steps -= 1
            if self.state.steps == 0:
                self.logger.info(
                    "Starting actual profiling for power limit %d W.",
                    self.state.current_power_limit // 1000,
                )
                self.state = Profiling(
                    current_power_limit=self.state.current_power_limit,
                    steps=self.profile_steps,
                )
                self.monitor.begin_window(
                    f"__PowerLimitOptimizer_{self.state.current_power_limit // 1000}",
                )

        elif isinstance(self.state, Profiling):
            self.state.steps -= 1
            if self.state.steps == 0:
                measurement = self.monitor.end_window(
                    f"__PowerLimitOptimizer_{self.state.current_power_limit // 1000}",
                )
                self.logger.info(
                    "Finished profiling for power limit %d W.",
                    self.state.current_power_limit // 1000,
                )
                self.measurements.append(
                    Measurement(
                        power_limit=self.state.current_power_limit // 1000,
                        energy=measurement.total_energy,
                        time=measurement.time,
                    )
                )
                # If we're done profiling all power limits, compute the optimal
                # power limit and transition to the Done state. Otherwise, move
                # on to the Warmup phase for the next power limit.
                current_power_limit_index = self.power_limits.index(
                    self.state.current_power_limit
                )
                if current_power_limit_index == len(self.power_limits) - 1:
                    self.state = Done(
                        optimal_power_limit=self._compute_optimal_power_limit(),
                    )
                    self._set_power_limit(self.state.optimal_power_limit)
                    self._save_profile()
                else:
                    next_power_limit = self.power_limits[current_power_limit_index + 1]
                    self.logger.info(
                        "Starting warmup for power limit %d W.",
                        next_power_limit // 1000,
                    )
                    self._set_power_limit(next_power_limit)
                    self.state = Warmup(
                        current_power_limit=next_power_limit,
                        steps=self.warmup_steps,
                    )

        elif isinstance(self.state, Done):
            pass

    def on_step_end(self) -> None:
        """Mark the end of a training step."""
        if isinstance(self.state, Ready):
            raise RuntimeError("on_step_begin() must be called before on_step_end().")

        elif isinstance(self.state, (Warmup, Profiling, Done)):
            pass

    def _set_power_limit(self, power_limit: int) -> None:
        """Set the power limit for all GPUs.

        Args:
            power_limit: The power limit to set, in milliWatts.
        """
        self.logger.info("Setting power limit to %d W.", power_limit // 1000)
        if self.current_power_limit == power_limit:
            return
        for handle in self.handles:
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, power_limit)
        self.current_power_limit = power_limit

    def _compute_optimal_power_limit(self) -> int:
        """Compute the optimal power limit in milliWatts.

        The optimal power limit is the one that minimizes the Zeus time-energy cost.
        """
        max_power = max(self.power_limits) // 1000 * len(self.monitor.gpu_indices)
        cost_map = {
            measurement.power_limit
            * 1000: zeus_cost(
                energy=measurement.energy,
                time=measurement.time,
                eta_knob=self.eta_knob,
                max_power=max_power,
            )
            for measurement in self.measurements
        }
        optimal_power_limit = min(cost_map, key=lambda x: cost_map[x])
        self.logger.info("Optimal power limit is %d W.", optimal_power_limit // 1000)
        return optimal_power_limit

    def _save_profile(self) -> None:
        """Save JIT profiling results and the optimal power limit to a JSON file."""
        if self.profile_path is None:
            return

        assert isinstance(self.state, Done)
        profile = {
            "measurements": [
                {
                    "power_limit": measurement.power_limit,
                    "energy": measurement.energy,
                    "time": measurement.time,
                }
                for measurement in self.measurements
            ],
        }
        with self.profile_path.open("w", encoding="utf-8") as f:
            json.dump(profile, f, indent=4)
        self.logger.info("JIT profiling results saved to '%s'.", str(self.profile_path))
