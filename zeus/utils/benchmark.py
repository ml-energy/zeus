"""Utility function for measuring the time and energy of a callable."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any

    from zeus.monitor.energy import ZeusMonitor, Measurement


def measure(
    fn: Callable[..., Any],
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    *,
    num_repeats: int = 1,
    zeus_monitor: ZeusMonitor | None = None,
) -> Measurement:
    """Measure the time and energy consumption of a function.

    This utility function provides a convenient way to benchmark a callable
    by measuring its execution time and energy consumption. It handles
    measurement window management and supports repeated execution for
    more accurate measurements.

    Example:
        ```python
        from zeus.monitor import ZeusMonitor
        from zeus.utils.benchmark import measure

        def my_training_step(model, data):
            return model(data)

        # Using an existing monitor
        monitor = ZeusMonitor(gpu_indices=[0])
        result = measure(
            my_training_step,
            args=[model, batch],
            num_repeats=10,
            zeus_monitor=monitor,
        )
        print(f"Total time: {result.time}s, Total energy: {result.total_energy}J")

        # Without providing a monitor (one will be created automatically)
        result = measure(lambda: time.sleep(1))
        ```

    Args:
        fn: The function to benchmark.
        args: Positional arguments to pass to the function.
        kwargs: Keyword arguments to pass to the function.
        num_repeats: Number of times to repeat the function execution. All
            repetitions are measured together in a single measurement window.
            Defaults to 1.
        zeus_monitor: An existing [`ZeusMonitor`][zeus.monitor.energy.ZeusMonitor]
            instance to use for measurement. If `None`, a new monitor will be
            created with default settings (monitoring all available GPUs).

    Returns:
        A [`Measurement`][zeus.monitor.energy.Measurement] object containing
            the total time elapsed and energy consumed across all repetitions.

    Raises:
        ValueError: If `num_repeats` is less than 1.

    Note:
        If the measurement duration is very short, the energy consumption may
        be reported as zero due to the GPU's energy counter update period.
        In such cases, consider increasing `num_repeats` or enabling
        `approx_instant_energy` on the `ZeusMonitor`.
    """
    # Import here to avoid circular imports
    from zeus.monitor.energy import ZeusMonitor as ZeusMonitorClass

    if num_repeats < 1:
        raise ValueError(f"num_repeats must be at least 1, got {num_repeats}")

    # Create a monitor if not provided
    monitor = ZeusMonitorClass() if zeus_monitor is None else zeus_monitor

    # Prepare arguments
    call_args: Sequence[Any] = args if args is not None else ()
    call_kwargs: Mapping[str, Any] = kwargs if kwargs is not None else {}

    # Use a unique window name
    window_name = "__zeus_measure_window__"

    # Start measurement
    monitor.begin_window(window_name)

    # Execute the function num_repeats times
    for _ in range(num_repeats):
        fn(*call_args, **call_kwargs)

    # End measurement and get result
    measurement = monitor.end_window(window_name)

    # Provide additional guidance if energy is zero
    if any(energy == 0.0 for energy in measurement.gpu_energy.values()):
        warnings.warn(
            "The energy consumption was measured as zero. The measurement "
            "duration may be too short for the GPU's energy counter update "
            "period. Consider increasing `num_repeats` or enabling "
            "`approx_instant_energy` on the `ZeusMonitor`.",
            stacklevel=2,
        )

    return measurement
