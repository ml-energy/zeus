"""Thermally stable energy profiling for GPU workloads.

The module provides functions to determine the best measurement and cooldown
durations that yield stable (low-variance) energy measurements for a
user-provided callable.

**Measurement duration sweep**: fixes cooldown_duration at the maximum of the
cooldown search range and sweeps measurement_duration.  Each configuration is
measured for `num_trials` trials; configurations whose energy standard
deviation falls below `variance_threshold` are considered valid.

**Cooldown duration sweep**: fixes measurement_duration at the maximum of the
measurement search range and sweeps cooldown_duration with the same validity
criterion.

Both durations can also be chosen manually and passed directly to
[`measure`][zeus.monitor.kernel_profiler.measure].

**Multi-GPU / distributed setting**: In a distributed setting each rank should
create its own [`ZeusMonitor`][zeus.monitor.energy.ZeusMonitor] with
``gpu_indices=[local_rank]`` and pass it to the profiling functions.  Every
rank executes the workload and measures energy on its local GPU.
[`all_reduce`][zeus.utils.framework.all_reduce] is used internally to
aggregate results across ranks (energy is summed, time takes the max across
ranks, and temperature is averaged).  When the distributed backend is not
initialized, ``all_reduce`` is a no-op and single-GPU behavior is preserved.
Only rank 0 prints progress and reports.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)

from zeus.monitor.energy import ZeusMonitor
from zeus.utils.framework import all_reduce, get_rank, get_world_size, sync_execution

logger = logging.getLogger(__name__)

_DEFAULT_SEARCH_RANGE = [float(x) for x in range(1, 11)]


def _is_rank_zero() -> bool:
    """Return True when not distributed or when this is rank 0."""
    return get_rank() == 0


@dataclass
class TrialResult:
    """Result of a single measurement trial.

    Attributes:
        energy_per_iter: Energy consumed per iteration (Joules).
        time_per_iter: Wall-clock time per iteration (seconds).
        total_energy: Total energy consumed during the measurement window (Joules).
        total_time: Total wall-clock time of the measurement window (seconds).
        iterations: Number of iterations executed in the window.
        temperature_before: GPU temperature (Celsius) before the measurement.
        temperature_after: GPU temperature (Celsius) after the measurement.
    """

    energy_per_iter: float
    time_per_iter: float
    total_energy: float
    total_time: float
    iterations: int
    temperature_before: float
    temperature_after: float


@dataclass
class SweepResult:
    """Result of sweeping one parameter value across multiple trials.

    Attributes:
        measurement_duration: The measurement duration used (seconds).
        cooldown_duration: The cooldown duration used (seconds).
        trials: Per-trial results.
        energy_mean: Mean `energy_per_iter` across trials.
        energy_std: Sample standard deviation of `energy_per_iter` across trials.
        avg_temperature_before: Average temperature before the measurement.
        avg_temperature_after: Average temperature after the measurement.
        avg_total_time: Mean total time across trials.
        avg_total_energy: Mean total energy across trials.
        is_valid: `True` when `energy_std < variance_threshold`.
    """

    measurement_duration: float
    cooldown_duration: float
    trials: list[TrialResult]
    energy_mean: float
    energy_std: float
    avg_temperature_before: float
    avg_temperature_after: float
    avg_total_time: float
    avg_total_energy: float
    is_valid: bool

    def __str__(self) -> str:
        """One-line summary without per-trial details."""
        tag = "VALID" if self.is_valid else "INVALID"
        return (
            f"measurement={self.measurement_duration:.1f}s  "
            f"cooldown={self.cooldown_duration:.1f}s  "
            f"mean={self.energy_mean:.4f} J  "
            f"std={self.energy_std:.4f} J  "
            f"temp_before={self.avg_temperature_before:.1f}\u00b0C  "
            f"temp_after={self.avg_temperature_after:.1f}\u00b0C  "
            f"[{tag}]"
        )


@dataclass
class SweepReport:
    """Full report from a measurement-duration or cooldown-duration sweep.

    Attributes:
        sweep_param: `(parameter_name, swept_values)`.
        fixed_param: `(parameter_name, fixed_value)`.
        entries: All sweep entries (one per swept value).
    """

    sweep_param: tuple[str, list[float]]
    fixed_param: tuple[str, float]
    entries: list[SweepResult]

    def __str__(self) -> str:
        """Multi-line summary: one line per swept value."""
        sweep_name, _ = self.sweep_param
        fixed_name, fixed_value = self.fixed_param
        lines = [f"Sweep {sweep_name} (fixed {fixed_name}={fixed_value:.1f}s):"]
        for e in self.entries:
            swept_value = getattr(e, sweep_name)
            tag = "VALID" if e.is_valid else "INVALID"
            lines.append(
                f"  {sweep_name}={swept_value:.1f}s  "
                f"mean={e.energy_mean:.4f} J  "
                f"std={e.energy_std:.4f} J  "
                f"temp_before={e.avg_temperature_before:.1f}\u00b0C  "
                f"temp_after={e.avg_temperature_after:.1f}\u00b0C  "
                f"[{tag}]"
            )
        return "\n".join(lines)


def _calibrate_iter_duration(
    target_function: Callable[..., Any],
    zeus_monitor: ZeusMonitor,
    warmup_iterations: int,
    calibration_iterations: int,
) -> float:
    """Warm up *target_function* and measure per-iteration execution time."""
    for _ in range(warmup_iterations):
        target_function()

    sync_execution(zeus_monitor.gpu_indices, sync_with=zeus_monitor.sync_with)
    start = time.monotonic()
    for _ in range(calibration_iterations):
        target_function()
    sync_execution(zeus_monitor.gpu_indices, sync_with=zeus_monitor.sync_with)
    elapsed = time.monotonic() - start

    iter_duration = elapsed / calibration_iterations
    [iter_duration] = all_reduce([iter_duration], "max")
    logger.info("Calibrated iteration duration: %.3f ms", iter_duration * 1000)
    return iter_duration


def _iterations_for(iter_duration: float, target_duration: float) -> int:
    """Return how many iterations fill *target_duration* seconds."""
    return max(1, int(target_duration / iter_duration))


def _read_avg_gpu_temperature(zeus_monitor: ZeusMonitor) -> float:
    """Return the mean GPU temperature (deg C) across all monitored GPUs."""
    temps = [zeus_monitor.gpus.get_gpu_temperature(idx) for idx in zeus_monitor.gpu_indices]
    return sum(temps) / len(temps) if temps else 0.0


def _run_trial(
    target_function: Callable[..., Any],
    zeus_monitor: ZeusMonitor,
    cooldown_duration: float,
    measurement_duration: float,
    warmup_iterations: int,
    iter_duration: float,
) -> TrialResult:
    """Execute one trial: cooldown -> warmup -> measure."""
    iterations = _iterations_for(iter_duration, measurement_duration)

    if cooldown_duration > 0:
        time.sleep(cooldown_duration)

    temperature_before = _read_avg_gpu_temperature(zeus_monitor)

    for _ in range(warmup_iterations):
        target_function()

    zeus_monitor.begin_window("_trial")
    for _ in range(iterations):
        target_function()
    result = zeus_monitor.end_window("_trial")

    temperature_after = _read_avg_gpu_temperature(zeus_monitor)

    [total_energy] = all_reduce([result.total_energy], "sum")
    [total_time] = all_reduce([result.time], "max")
    [temp_before_sum] = all_reduce([temperature_before], "sum")
    [temp_after_sum] = all_reduce([temperature_after], "sum")
    world_size = get_world_size()

    return TrialResult(
        energy_per_iter=total_energy / iterations,
        time_per_iter=total_time / iterations,
        total_energy=total_energy,
        total_time=total_time,
        iterations=iterations,
        temperature_before=temp_before_sum / world_size,
        temperature_after=temp_after_sum / world_size,
    )


def _build_sweep_result(
    measurement_duration: float,
    cooldown_duration: float,
    trials: list[TrialResult],
    variance_threshold: float,
) -> SweepResult:
    """Aggregate trial results into a [`SweepResult`][zeus.monitor.kernel_profiler.SweepResult]."""
    energies = [t.energy_per_iter for t in trials]
    n = len(trials)
    e_std = statistics.stdev(energies) if n >= 2 else 0.0
    return SweepResult(
        measurement_duration=measurement_duration,
        cooldown_duration=cooldown_duration,
        trials=trials,
        energy_mean=statistics.mean(energies),
        energy_std=e_std,
        avg_temperature_before=sum(t.temperature_before for t in trials) / n,
        avg_temperature_after=sum(t.temperature_after for t in trials) / n,
        avg_total_time=sum(t.total_time for t in trials) / n,
        avg_total_energy=sum(t.total_energy for t in trials) / n,
        is_valid=e_std < variance_threshold,
    )


def _sweep(
    target_function: Callable[..., Any],
    zeus_monitor: ZeusMonitor,
    sweep_values: list[float],
    fixed_value: float,
    sweep_type: str,
    num_trials: int,
    variance_threshold: float,
    warmup_iterations: int,
    iter_duration: float,
) -> SweepReport:
    """Run a parameter sweep and return a [`SweepReport`][zeus.monitor.kernel_profiler.SweepReport]."""
    is_cooldown_sweep = sweep_type == "cooldown_duration"
    show_progress = _is_rank_zero()

    entries: list[SweepResult] = []
    progress_ctx = (
        Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
        if show_progress
        else None
    )
    if progress_ctx is not None:
        progress_ctx.start()

    try:
        for val in sweep_values:
            cooldown_dur = val if is_cooldown_sweep else fixed_value
            measure_dur = fixed_value if is_cooldown_sweep else val

            task_id = (
                progress_ctx.add_task(f"{sweep_type}={val:.1f}s", total=num_trials)
                if progress_ctx is not None
                else None
            )
            trials: list[TrialResult] = []
            for _ in range(num_trials):
                trial = _run_trial(
                    target_function=target_function,
                    zeus_monitor=zeus_monitor,
                    cooldown_duration=cooldown_dur,
                    measurement_duration=measure_dur,
                    warmup_iterations=warmup_iterations,
                    iter_duration=iter_duration,
                )
                trials.append(trial)
                if progress_ctx is not None and task_id is not None:
                    progress_ctx.advance(task_id)

            entry = _build_sweep_result(measure_dur, cooldown_dur, trials, variance_threshold)
            entries.append(entry)
    finally:
        if progress_ctx is not None:
            progress_ctx.stop()

    swept_name = "cooldown_duration" if is_cooldown_sweep else "measurement_duration"
    fixed_name = "measurement_duration" if is_cooldown_sweep else "cooldown_duration"
    return SweepReport(
        sweep_param=(swept_name, sweep_values),
        fixed_param=(fixed_name, fixed_value),
        entries=entries,
    )


def profile_measurement_duration(
    target_function: Callable[..., Any],
    zeus_monitor: ZeusMonitor,
    measurement_duration_search_range: list[float] | None = None,
    fixed_cooldown_duration: float = 10.0,
    num_trials: int = 10,
    variance_threshold: float = 0.01,
    warmup_iterations: int = 10,
    calibration_iterations: int = 100,
    iter_duration: float | None = None,
) -> SweepReport:
    """Sweep measurement durations and return a [`SweepReport`][zeus.monitor.kernel_profiler.SweepReport].

    Args:
        target_function: Callable to profile (invoked with no arguments).
        zeus_monitor: [`ZeusMonitor`][zeus.monitor.energy.ZeusMonitor] instance.
        measurement_duration_search_range: Durations (seconds) to sweep.
            Defaults to `[1.0, 2.0, ..., 10.0]`.
        fixed_cooldown_duration: Cooldown held constant during the sweep.
        num_trials: Repeated trials per sweep point.
        variance_threshold: Maximum acceptable `energy_std` (Joules) for a
            duration to be considered valid.
        warmup_iterations: Warm-up iterations before each measurement.
        calibration_iterations: Iterations used to estimate per-iteration time.
        iter_duration: Pre-calibrated iteration duration (seconds).  If `None`,
            calibration runs automatically.
    """
    search_range = measurement_duration_search_range or list(_DEFAULT_SEARCH_RANGE)
    if iter_duration is None:
        iter_duration = _calibrate_iter_duration(
            target_function, zeus_monitor, warmup_iterations, calibration_iterations
        )

    if _is_rank_zero():
        print(f"=== Sweeping measurement_duration (cooldown fixed at {fixed_cooldown_duration:.1f}s) ===")
    report = _sweep(
        target_function=target_function,
        zeus_monitor=zeus_monitor,
        sweep_values=search_range,
        fixed_value=fixed_cooldown_duration,
        sweep_type="measurement_duration",
        num_trials=num_trials,
        variance_threshold=variance_threshold,
        warmup_iterations=warmup_iterations,
        iter_duration=iter_duration,
    )
    if _is_rank_zero():
        print(report)
    return report


def profile_cooldown_duration(
    target_function: Callable[..., Any],
    zeus_monitor: ZeusMonitor,
    cooldown_duration_search_range: list[float] | None = None,
    fixed_measurement_duration: float = 10.0,
    num_trials: int = 10,
    variance_threshold: float = 0.01,
    warmup_iterations: int = 10,
    calibration_iterations: int = 100,
    iter_duration: float | None = None,
) -> SweepReport:
    """Sweep cooldown durations and return a [`SweepReport`][zeus.monitor.kernel_profiler.SweepReport].

    Args:
        target_function: Callable to profile (invoked with no arguments).
        zeus_monitor: [`ZeusMonitor`][zeus.monitor.energy.ZeusMonitor] instance.
        cooldown_duration_search_range: Durations (seconds) to sweep.
            Defaults to `[1.0, 2.0, ..., 10.0]`.
        fixed_measurement_duration: Measurement duration held constant during
            the sweep.
        num_trials: Repeated trials per sweep point.
        variance_threshold: Maximum acceptable `energy_std` (Joules) for a
            duration to be considered valid.
        warmup_iterations: Warm-up iterations before each measurement.
        calibration_iterations: Iterations used to estimate per-iteration time.
        iter_duration: Pre-calibrated iteration duration (seconds).  If `None`,
            calibration runs automatically.
    """
    search_range = cooldown_duration_search_range or list(_DEFAULT_SEARCH_RANGE)
    if iter_duration is None:
        iter_duration = _calibrate_iter_duration(
            target_function, zeus_monitor, warmup_iterations, calibration_iterations
        )

    if _is_rank_zero():
        print(f"=== Sweeping cooldown_duration (measurement fixed at {fixed_measurement_duration:.1f}s) ===")
    report = _sweep(
        target_function=target_function,
        zeus_monitor=zeus_monitor,
        sweep_values=search_range,
        fixed_value=fixed_measurement_duration,
        sweep_type="cooldown_duration",
        num_trials=num_trials,
        variance_threshold=variance_threshold,
        warmup_iterations=warmup_iterations,
        iter_duration=iter_duration,
    )
    if _is_rank_zero():
        print(report)
    return report


def profile_parameters(
    target_function: Callable[..., Any],
    zeus_monitor: ZeusMonitor,
    measurement_duration_search_range: list[float] | None = None,
    cooldown_duration_search_range: list[float] | None = None,
    num_trials: int = 10,
    variance_threshold: float = 0.01,
    warmup_iterations: int = 10,
    calibration_iterations: int = 100,
) -> tuple[SweepReport, SweepReport]:
    """Auto-profile both measurement and cooldown durations.

    Performs two sequential sweeps:

    1. **Measurement duration sweep** -- cooldown is fixed at the *maximum*
       of `cooldown_duration_search_range`.
    2. **Cooldown duration sweep** -- measurement duration is fixed at the
       *maximum* of `measurement_duration_search_range`.

    Args:
        target_function: Callable to profile (invoked with no arguments).
        zeus_monitor: [`ZeusMonitor`][zeus.monitor.energy.ZeusMonitor] instance.
        measurement_duration_search_range: Durations (seconds) to sweep.
            Defaults to `[1.0, 2.0, ..., 10.0]`.
        cooldown_duration_search_range: Durations (seconds) to sweep.
            Defaults to `[1.0, 2.0, ..., 10.0]`.
        num_trials: Repeated trials per sweep point.
        variance_threshold: Maximum acceptable `energy_std` (Joules) for a
            duration to be considered valid.
        warmup_iterations: Warm-up iterations before each measurement.
        calibration_iterations: Iterations used to estimate per-iteration time.

    Returns:
        `(measurement_sweep_report, cooldown_sweep_report)`
    """
    m_range = measurement_duration_search_range or list(_DEFAULT_SEARCH_RANGE)
    c_range = cooldown_duration_search_range or list(_DEFAULT_SEARCH_RANGE)

    iter_dur = _calibrate_iter_duration(target_function, zeus_monitor, warmup_iterations, calibration_iterations)

    measurement_report = profile_measurement_duration(
        target_function=target_function,
        zeus_monitor=zeus_monitor,
        measurement_duration_search_range=m_range,
        fixed_cooldown_duration=max(c_range),
        num_trials=num_trials,
        variance_threshold=variance_threshold,
        warmup_iterations=warmup_iterations,
        iter_duration=iter_dur,
    )

    cooldown_report = profile_cooldown_duration(
        target_function=target_function,
        zeus_monitor=zeus_monitor,
        cooldown_duration_search_range=c_range,
        fixed_measurement_duration=max(m_range),
        num_trials=num_trials,
        variance_threshold=variance_threshold,
        warmup_iterations=warmup_iterations,
        iter_duration=iter_dur,
    )

    return measurement_report, cooldown_report


def measure(
    target_function: Callable[..., Any],
    zeus_monitor: ZeusMonitor,
    measurement_duration: float,
    cooldown_duration: float,
    warmup_iterations: int = 10,
    calibration_iterations: int = 100,
    iter_duration: float | None = None,
) -> TrialResult:
    """Run a single energy measurement trial.

    Args:
        target_function: Callable to profile (invoked with no arguments).
        zeus_monitor: [`ZeusMonitor`][zeus.monitor.energy.ZeusMonitor] instance.
        measurement_duration: Target measurement window length (seconds).
        cooldown_duration: Idle time before the measurement (seconds).
        warmup_iterations: Warm-up iterations before the measurement.
        calibration_iterations: Iterations used to estimate per-iteration time
            (only used when `iter_duration` is `None`).
        iter_duration: Iteration duration of target_function (seconds).  If `None`,
            calibration runs automatically.
    """
    if iter_duration is None:
        iter_duration = _calibrate_iter_duration(
            target_function, zeus_monitor, warmup_iterations, calibration_iterations
        )

    return _run_trial(
        target_function=target_function,
        zeus_monitor=zeus_monitor,
        cooldown_duration=cooldown_duration,
        measurement_duration=measurement_duration,
        warmup_iterations=warmup_iterations,
        iter_duration=iter_duration,
    )
