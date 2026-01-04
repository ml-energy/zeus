"""Example of using the `measure` utility function.

The `measure` function provides a simple way to measure the time and energy
consumption of any callable, without needing to manually manage measurement
windows.
"""

import time

from zeus.monitor import ZeusMonitor
from zeus.utils import measure


def workload(duration: float, iterations: int) -> int:
    """A sample workload that sleeps and does some computation."""
    time.sleep(duration)
    result = 0
    for i in range(iterations):
        result += i * i
    return result


def main():
    """Run the example demonstrating the measure utility function."""
    # Create a ZeusMonitor to measure GPU energy.
    # You can also omit this and let `measure` create one automatically.
    monitor = ZeusMonitor(gpu_indices=[0])

    # Basic usage: measure a simple function
    print("=" * 50)
    print("Basic usage: measuring a simple function")
    print("=" * 50)
    measurement = measure(lambda: time.sleep(0.5), zeus_monitor=monitor)
    print(f"Time: {measurement.time:.3f} seconds")
    print(f"GPU energy: {measurement.total_energy:.3f} Joules")
    print()

    # Measure a function with arguments
    print("=" * 50)
    print("Measuring a function with arguments")
    print("=" * 50)
    measurement = measure(
        workload,
        args=[0.1],  # duration
        kwargs={"iterations": 100000},
        zeus_monitor=monitor,
    )
    print(f"Time: {measurement.time:.3f} seconds")
    print(f"GPU energy: {measurement.total_energy:.3f} Joules")
    print()

    # Use num_repeats for more accurate measurements
    print("=" * 50)
    print("Using num_repeats for repeated execution")
    print("=" * 50)
    measurement = measure(
        workload,
        args=[0.05, 10000],
        num_repeats=5,
        zeus_monitor=monitor,
    )
    print(f"Total time for 5 repeats: {measurement.time:.3f} seconds")
    print(f"Average time per call: {measurement.time / 5:.3f} seconds")
    print(f"Total GPU energy: {measurement.total_energy:.3f} Joules")
    print()

    # Without providing a monitor (one will be created automatically)
    print("=" * 50)
    print("Without providing a monitor")
    print("=" * 50)
    measurement = measure(lambda: time.sleep(0.2))
    print(f"Time: {measurement.time:.3f} seconds")
    print(f"GPU energy: {measurement.total_energy:.3f} Joules")


if __name__ == "__main__":
    main()
