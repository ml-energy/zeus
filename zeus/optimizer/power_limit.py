"""Optimizers that select the optimum power limit.

This module contains the following pieces:

- [`GlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.GlobalPowerLimitOptimizer]
  is the main class that implements the state machine
  and the logic for profiling power limits and selecting
  the optimum power limit.
- [`PowerLimitMeasurement`][zeus.optimizer.power_limit.PowerLimitMeasurement] and various
  state classes are helpers that support the state machine.
- [`OptimumSelector`][zeus.optimizer.power_limit.OptimumSelector]
  is an abstract base class for selecting the optimum power limit
  from a list of power limit profiling results. There are concrete classes
  that implement different selection strategies, like
  [minimizing energy][zeus.optimizer.power_limit.Energy],
  [minimizing time][zeus.optimizer.power_limit.Time],
  [minimizing the Zeus time-energy cost][zeus.optimizer.power_limit.ZeusCost],
  or [selecting the lowest power limit that meets the given maximum training time slowdown factor][zeus.optimizer.power_limit.MaxSlowdownConstraint].
- [`HFGlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.HFGlobalPowerLimitOptimizer]
  is a wrapper for the Hugging Face `TrainerCallback` class that uses `GlobalPowerLimitOptimizer`.
"""

from __future__ import annotations

import atexit
from pathlib import Path
from abc import ABC, abstractmethod

from zeus.callback import Callback
from zeus.monitor import ZeusMonitor
from zeus.utils.framework import all_reduce, is_distributed
from zeus.utils.logging import get_logger
from zeus.utils.metric import zeus_cost
from zeus.utils.pydantic_v1 import BaseModel, PositiveInt, PositiveFloat
from zeus.device import get_gpus
from zeus.device.gpu import ZeusGPUNoPermissionError

from typing import TYPE_CHECKING


class OptimumSelector(ABC):
    """Base class for optimum power limit selectors."""

    @abstractmethod
    def select(self, measurements: list[PowerLimitMeasurement]) -> int:
        """Select the optimal power limit (W) from measurements."""


class Energy(OptimumSelector):
    """Selects the power limit that minimizes energy consumption."""

    def select(self, measurements: list[PowerLimitMeasurement]) -> int:
        """Select the optimal power limit (W) from measurements."""
        return min(measurements, key=lambda x: x.energy).power_limit


class Time(OptimumSelector):
    """Selects the power limit that minimizes training time.

    This may not necessarily choose the maximum power limit, as time profiling
    results can be slightly noisy. However, we believe that's actually better
    because it means that training time is very similar among higher power limits,
    but lower power limit will consume less power.
    """

    def select(self, measurements: list[PowerLimitMeasurement]) -> int:
        """Select the optimal power limit (W) from measurements."""
        return min(measurements, key=lambda x: x.time).power_limit


class ZeusCost(OptimumSelector):
    r"""Selects the power limit that minimizes a linear Zeus time-energy cost function.

    Cost function is $\eta \cdot \text{Energy} + (1 - \eta) \cdot \text{MaxPower} \cdot \text{Time}$.
    """

    def __init__(self, eta_knob: float, world_size: int = 1) -> None:
        r"""Initialize the selector.

        Args:
            eta_knob: The $0 \le \eta \le 1$ knob for the Zeus time-energy cost function.
            world_size: The number of GPUs in the training job. Defaults to 1.
        """
        if eta_knob < 0 or eta_knob > 1:
            raise ValueError("eta_knob must be between 0 and 1, inclusive both sides.")
        if world_size < 1:
            raise ValueError("world_size must be greater than or equal to 1.")

        self.eta_knob = eta_knob
        self.world_size = world_size

    def select(self, measurements: list[PowerLimitMeasurement]) -> int:
        """Select the optimal power limit (W) from measurements."""
        max_power = (
            max(measurement.power_limit for measurement in measurements)
            * self.world_size
        )
        zeus_cost_map = {
            measurement.power_limit: zeus_cost(
                energy=measurement.energy,
                time=measurement.time,
                eta_knob=self.eta_knob,
                max_power=max_power,
            )
            for measurement in measurements
        }
        return min(zeus_cost_map, key=lambda x: zeus_cost_map[x])


class MaxSlowdownConstraint(OptimumSelector):
    """Selects the minumum power limit that does not slow down training by more than the given factor."""

    def __init__(self, factor: float) -> None:
        """Initialize the selector.

        Args:
            factor: The maximum allowed slowdown factor. Greater than or equal to 1.0.
        """
        if factor < 1.0:
            raise ValueError(
                f"max_slowdown_factor must be greater than or equal to 1.0. Got {factor}.",
            )

        self.factor = factor

    def select(self, measurements: list[PowerLimitMeasurement]) -> int:
        """Select the optimal power limit (W) from measurements."""
        feasible_power_limits = []
        max_power = max(measurement.power_limit for measurement in measurements)
        shortest_time = next(
            measurement.time
            for measurement in measurements
            if measurement.power_limit == max_power
        )
        for measurement in measurements:
            if measurement.time <= self.factor * shortest_time:
                feasible_power_limits.append(measurement.power_limit)
        return min(feasible_power_limits)


class Ready(BaseModel):
    """State for when we are ready to start measuring the next power limit.

    Initial state of the state machine if no previous profiling results were given.
    `Ready` -> `Warmup` after `step`'th `on_step_begin`.
    """

    next_power_limit: PositiveInt
    steps: PositiveInt


class Warmup(BaseModel):
    """State for when we are warming up for a power limit.

    `Warmup` -> `Profiling` on the `steps`'th `on_step_begin`.
    `Warmup` -> `Ready` on `on_epoch_end` before `steps`'th `on_step_begin`.
    """

    current_power_limit: PositiveInt
    steps: PositiveInt


class Profiling(BaseModel):
    """State for when we are profiling a power limit.

    `Profiling` -> `Warmup` after `steps`'th `on_step_begin` and
        there are still power limits left to profile.
    `Profiling` -> `Done` after `steps`'th `on_step_begin` and
        there are no more power limits left to profile.
    `Profiling` -> `Ready` on `on_epoch_end` before `steps`'th `on_step_begin`.
    """

    current_power_limit: PositiveInt
    steps: PositiveInt


class Done(BaseModel):
    """State for when we are done profiling all power limits.

    Initial state of the state machine if previous profiling results were given.
    Final state of the state machine in any case.
    """

    optimal_power_limit: PositiveInt


class PowerLimitMeasurement(BaseModel):
    """POD for GPU energy and time measurements for one power limit (W)."""

    power_limit: PositiveInt  # In Watts.
    energy: PositiveFloat
    time: PositiveFloat


class _PowerLimitMeasurementList(BaseModel):
    """Proxy class to save and load a list of `PowerLimitMeasurement`s."""

    measurements: list[PowerLimitMeasurement]


class GlobalPowerLimitOptimizer(Callback):
    """Optimizer for the power limit knob.

    This optimizer uses the JIT profiling log to determine the optimal power limit.

    ## Usage with distributed data parallelism

    The global power limit optimizer expects one process to control each GPU used for training.
    For instance, `torchrun` will automatically spawn one process for each GPU on the node.
    Correspondingly, the [`ZeusMonitor`][zeus.monitor.energy.ZeusMonitor] instance passed in
    should be monitoring **one GPU**: the one being managed by the current process. The index of
    this GPU would typically match the local rank of the process. In the case of PyTorch, users would have
    called `torch.cuda.set_device` early on, so `torch.cuda.current_device` will give you the GPU index.
    `GlobalPowerLimitOptimizer` will internally do an AllReduce across all GPUs to aggregate
    time and energy measurements, and then select the globally optimal power limit.


    ```python
    monitor = ZeusMonitor(gpu_indices=[local_rank])  # pass in local rank to gpu_indices.
    plo = GlobalPowerLimitOptimizer(monitor)
    ```
    """

    def __init__(
        self,
        monitor: ZeusMonitor,
        optimum_selector: OptimumSelector | None = None,
        wait_steps: int = 1,
        warmup_steps: int = 10,
        profile_steps: int = 40,
        pl_step: int = 25,
        profile_path: str | Path | None = None,
    ) -> None:
        r"""Initialize the optimizer.

        GPU indices to profile and optimize for are taken from `monitor.gpu_indices`.

        Args:
            monitor: `ZeusMonitor` instance used to profile GPU time and energy consumption.
            optimum_selector: The optimum selector to use. If not given, use `ZeusCost` with \eta=0.5.
            wait_steps: Number of steps to pass by before doing anything at the beginning.
                Useful if you have something like `torch.backends.cudnn.benchmark=True`,
                because the first iteration won't be representative of the rest of the iterations.
            warmup_steps: Number of warmup iterations for each power limit.
            profile_steps: Number of profie iterations for each power limit.
            pl_step: The stride between power limits to explore, in unites of Watts.
            profile_path: If the path points to an existing file, load the profile from the file
                and do not run any profiling. If the path points to a non-existing file, profile
                and save the profile to the file. If `None`, do not save or load any profile.
        """
        # Sanity checks.
        if wait_steps < 0:
            raise ValueError("wait_steps must be non-negative.")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if profile_steps <= 0:
            raise ValueError("profile_steps must be positive.")
        if pl_step <= 0:
            raise ValueError("pl_step must be positive.")

        self.monitor = monitor
        self.optimum_selector = optimum_selector or ZeusCost(
            eta_knob=0.5,
            world_size=len(monitor.gpu_indices),
        )
        self.warmup_steps = warmup_steps
        self.profile_steps = profile_steps
        self.pl_step = pl_step * 1000  # Internally, we use milliWatts.
        self.profile_path = (
            Path(profile_path) if isinstance(profile_path, str) else profile_path
        )

        # Setup logging.
        self.logger = get_logger(type(self).__name__)

        gpus = get_gpus(ensure_homogeneous=True)

        # Warn if distributed training is enabled with multiple GPUs monitored.
        if is_distributed() and len(monitor.gpu_indices) > 1:
            self.logger.warning(
                "Distributed training is enabled with %d GPUs monitored. "
                "For distributed training, it is recommended to monitor only one GPU per `ZeusMonitor` instance "
                "since `GlobalPowerLimitOptimizer` performs an all-reduce operation internally over all devices.",
                len(monitor.gpu_indices),
            )

        # Set the range of power limits to explore.
        # Assert that supported power limits ranges are uniform across GPUs.
        pls = []
        for index in monitor.gpu_indices:
            pls.append(gpus.getPowerManagementLimitConstraints(index))
        if not all(pls[0] == pl for pl in pls):
            raise ValueError("Power limits ranges are not uniform across GPUs.")
        self.power_limits = list(range(pls[0][1], pls[0][0] - 1, -self.pl_step))

        # Turn on persistence mode and set to the highest power limit.
        try:
            for index in monitor.gpu_indices:
                gpus.setPersistenceMode(index, enabled=True)
        except ZeusGPUNoPermissionError as ze:
            raise RuntimeError(
                "SYS_ADMIN capability is required to modify GPU power limits. See "
                "https://ml.energy/zeus/getting_started/#system-privileges "
                "for more information."
            ) from ze
        self.current_power_limit = 0

        # Store `Measurement` objects in a list, one for each power limit.
        self.measurements: list[PowerLimitMeasurement] = []

        # State for the profiler state machine.
        self.state: Ready | Warmup | Profiling | Done

        # Initialize JIT profiling states.
        if self.profile_path is None:
            self.logger.info("JIT profiling enabled.")
            self.logger.info("Will wait %d step(s) before profiling.", wait_steps)
            self.state = Ready(
                next_power_limit=self.power_limits[0], steps=wait_steps + 1
            )
            self.logger.info("Set power limit to the maximum before starting.")
            self._set_power_limit(max(self.power_limits))
        elif not self.profile_path.exists():
            self.logger.info(
                "JIT Profiling enabled. Profile will be saved to '%s'.",
                str(self.profile_path),
            )
            self.logger.info("Will wait %d step(s) before profiling.", wait_steps)
            self.state = Ready(
                next_power_limit=self.power_limits[0], steps=wait_steps + 1
            )
            self.logger.info("Set power limit to the maximum before starting.")
            self._set_power_limit(max(self.power_limits))
        else:
            self.measurements = _PowerLimitMeasurementList.parse_file(
                self.profile_path,
            ).measurements
            # self.measurements = _PowerLimitMeasurementList.model_validate_json(
            #     open(self.profile_path).read(),
            #     strict=True,
            # ).measurements
            self.logger.info(
                "Loaded previous profiling results from '%s'.", str(self.profile_path)
            )
            optimal_power_limit = self._compute_optimal_power_limit()
            self.logger.info(
                "Optimal power limit is %d W.", optimal_power_limit // 1000
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
                    f"__GlobalPowerLimitOptimizer_{self.state.current_power_limit // 1000}",
                    cancel=True,
                )
            self.state = Ready(next_power_limit=self.state.current_power_limit, steps=1)
            self._set_power_limit(max(self.power_limits))

        elif isinstance(self.state, Done):
            pass

    def on_step_begin(self) -> None:
        """Mark the beginning of a training step."""
        if isinstance(self.state, Ready):
            self.state.steps -= 1
            if self.state.steps == 0:
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
                    f"__GlobalPowerLimitOptimizer_{self.state.current_power_limit // 1000}",
                )

        elif isinstance(self.state, Profiling):
            self.state.steps -= 1
            if self.state.steps == 0:
                measurement = self.monitor.end_window(
                    f"__GlobalPowerLimitOptimizer_{self.state.current_power_limit // 1000}",
                )
                self.logger.info(
                    "Finished profiling for power limit %d W.",
                    self.state.current_power_limit // 1000,
                )

                self.measurements.append(
                    PowerLimitMeasurement(
                        power_limit=self.state.current_power_limit // 1000,
                        energy=sum(
                            all_reduce(
                                list(measurement.gpu_energy.values()), operation="sum"
                            )
                        ),
                        time=max(all_reduce([measurement.time], operation="max")),
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

    def _set_power_limit(self, power_limit: int) -> None:
        """Set the power limit for all GPUs.

        Args:
            power_limit: The power limit to set, in milliWatts.
        """
        gpus = get_gpus()
        self.logger.info("Setting power limit to %d W.", power_limit // 1000)
        if self.current_power_limit == power_limit:
            return
        for index in self.monitor.gpu_indices:
            gpus.setPowerManagementLimit(index, power_limit)
        self.current_power_limit = power_limit

    def _compute_optimal_power_limit(self) -> int:
        """Compute the optimal power limit in milliWatts."""
        optimal_power_limit = self.optimum_selector.select(self.measurements) * 1000
        self.logger.info("Optimal power limit is %d W.", optimal_power_limit // 1000)
        return optimal_power_limit

    def _save_profile(self) -> None:
        """Save JIT profiling results and the optimal power limit to a JSON file."""
        if self.profile_path is None:
            return

        assert isinstance(self.state, Done)
        with self.profile_path.open("w", encoding="utf-8") as f:
            f.write(
                _PowerLimitMeasurementList(measurements=self.measurements).json(
                    indent=4
                ),
            )
        self.logger.info("JIT profiling results saved to '%s'.", str(self.profile_path))


# Only import HuggingFace Classes when type checking, to avoid hard dependency on HuggingFace Transformers
if TYPE_CHECKING:
    from transformers import (
        TrainingArguments,
        TrainerState,
        TrainerControl,
        PreTrainedModel,
    )

try:
    from transformers import TrainerCallback

    transformers_available = True
except ModuleNotFoundError:
    transformers_available = False
    TrainerCallback = object  # Fallback base class


class HFGlobalPowerLimitOptimizer(TrainerCallback):  # type: ignore
    """[Wrapped for Hugging Face Trainer Callback] Optimizer for the power limit knob.

    This optimizer uses the JIT profiling log to determine the optimal power limit.
    See [`GlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.GlobalPowerLimitOptimizer]
    for the underlying optimizer implementation.
    """

    def __init__(
        self,
        monitor: ZeusMonitor,
        optimum_selector: OptimumSelector | None = None,
        wait_steps: int = 1,
        warmup_steps: int = 10,
        profile_steps: int = 40,
        pl_step: int = 25,
        profile_path: str | Path | None = None,
    ) -> None:
        r"""Initialize the optimizer.

        GPU indices to profile and optimize for are taken from `monitor.gpu_indices`.

        Args:
            monitor: `ZeusMonitor` instance used to profile GPU time and energy consumption.
            optimum_selector: The optimum selector to use. If not given, use `ZeusCost` with \eta=0.5.
            wait_steps: Number of steps to pass by before doing anything at the beginning.
                Useful if you have something like `torch.backends.cudnn.benchmark=True`,
                because the first iteration won't be representative of the rest of the iterations.
            warmup_steps: Number of warmup iterations for each power limit.
            profile_steps: Number of profie iterations for each power limit.
            pl_step: The stride between power limits to explore, in unites of Watts.
            profile_path: If the path points to an existing file, load the profile from the file
                and do not run any profiling. If the path points to a non-existing file, profile
                and save the profile to the file. If `None`, do not save or load any profile.
        """
        if not transformers_available:
            raise ImportError(
                "The transformers package is not installed. Please install it to use the HFGlobalPowerLimitOptimizer."
            )

        self.optimizer = GlobalPowerLimitOptimizer(
            monitor=monitor,
            optimum_selector=optimum_selector,
            wait_steps=wait_steps,
            warmup_steps=warmup_steps,
            profile_steps=profile_steps,
            pl_step=pl_step,
            profile_path=profile_path,
        )

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel,
        **kwargs,
    ) -> None:
        """Mark the end of a training epoch."""
        self.optimizer.on_epoch_end()

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel,
        **kwargs,
    ) -> None:
        """Mark the beginning of a training step."""
        self.optimizer.on_step_begin()
