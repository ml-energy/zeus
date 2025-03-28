"""Zeus Monitor CLI.

The CLI supports both power monitoring and energy measurement.
"""

from __future__ import annotations

import time
import argparse
from datetime import datetime

import rich

from zeus.monitor.energy import ZeusMonitor
from zeus.monitor.power import PowerMonitor


def energy(gpu_indices: list[int] | None = None) -> None:
    """Measure the time and energy of GPUs using the ZeusMonitor.

    Args:
        gpu_indices: Indices of GPUs to monitor. Not ommitted, all GPUs will be monitored.
    """
    monitor = ZeusMonitor(gpu_indices)
    monitor.begin_window("zeus.monitor.energy")

    try:
        # "Forever"
        time.sleep(365 * 24 * 60 * 60)
    except KeyboardInterrupt:
        energy = monitor.end_window("zeus.monitor.energy")
        if energy is not None:
            rich.print("Total energy (J):", energy)


def power(
    gpu_indices: list[int] | None = None,
    update_period: float = 1.0,
) -> None:
    """Monitor the power consumption of GPUs during the duration of the CLI program.

    Args:
        gpu_indices: Indices of GPUs to monitor. Not ommitted, all GPUs will be monitored.
        update_period: The time between power measurements in seconds.
    """
    monitor = PowerMonitor(gpu_indices=gpu_indices, update_period=update_period)
    start_time = time.time()
    update_period = monitor.update_period

    def map_gpu_index_to_name(measurements: dict[int, float]) -> dict[str, float]:
        return {f"GPU{k}": v for k, v in measurements.items()}

    try:
        while True:
            time.sleep(update_period)
            power = monitor.get_power()
            if power is None:
                continue
            rich.print(datetime.now(), map_gpu_index_to_name(power))
    except KeyboardInterrupt:
        end_time = time.time()
        rich.print("\nTotal time (s):", end_time - start_time)
        energy = monitor.get_energy(start_time, end_time)
        if energy is not None:
            rich.print("Total energy (J):", map_gpu_index_to_name(energy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m zeus.monitor",
        description="Zeus Monitor CLI",
    )

    # Subcommands for energy and power
    subparsers = parser.add_subparsers(
        dest="subcommand",
        required=True,
        help="The subcommand to run. See `zeus monitor <subcommand> --help` for more information.",
    )
    energy_parser = subparsers.add_parser(
        "energy",
        help="Measure the time and energy consumption of specified GPU indices.",
    )
    power_parser = subparsers.add_parser(
        "power",
        help="Monitor the power consumption of specified GPU indices, and compute total energy on exit (CTRL+C).",
    )

    # Arguments for energy
    energy_parser.add_argument(
        "--gpu-indices",
        nargs="+",
        type=int,
        help="Indices of GPUs to monitor. If omitted, all GPUs will be monitored.",
    )

    # Arguments for power
    power_parser.add_argument(
        "--gpu-indices",
        nargs="+",
        type=int,
        help="Indices of GPUs to monitor. If omitted, all GPUs will be monitored.",
    )
    power_parser.add_argument(
        "--update-period",
        type=float,
        default=1.0,
        help="The time between power measurements in seconds.",
    )

    args = parser.parse_args()

    # Dispatch to the appropriate subcommand
    if args.subcommand == "energy":
        energy(args.gpu_indices)
    elif args.subcommand == "power":
        power(args.gpu_indices, args.update_period)
    else:
        raise ValueError(f"Unknown subcommand: {args.subcommand}")
