"""Example script for querying CPU and GPU power timelines with PowerMonitor."""

from __future__ import annotations

import argparse
import json
import time

from zeus.monitor.power import PowerDomain, PowerMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="How long to sample power timelines, in seconds.",
    )
    parser.add_argument(
        "--update-period",
        type=float,
        default=0.1,
        help="Polling interval for PowerMonitor, in seconds.",
    )
    parser.add_argument(
        "--cpu-index",
        type=int,
        default=None,
        help="Optional CPU package index to query. Defaults to all monitored packages.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=None,
        help="Optional GPU index to query. Defaults to all monitored GPUs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()
    monitor = PowerMonitor(cpu_indices=None, update_period=args.update_period)

    try:
        time.sleep(args.duration)
        end_time = time.time()

        timelines = monitor.get_all_power_timelines(
            gpu_index=args.gpu_index,
            cpu_index=args.cpu_index,
            start_time=start_time,
            end_time=end_time,
        )

        cpu_package_timeline = monitor.get_power_timeline(
            PowerDomain.CPU_PACKAGE_AVERAGE,
            start_time=start_time,
            end_time=end_time,
            cpu_index=args.cpu_index,
        )

        result = {
            "window": {
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": args.duration,
            },
            "cpu_package_average": cpu_package_timeline,
            "all_power_timelines": timelines,
        }

        if PowerDomain.CPU_DRAM_AVERAGE.value in timelines:
            result["cpu_dram_average"] = monitor.get_power_timeline(
                PowerDomain.CPU_DRAM_AVERAGE,
                start_time=start_time,
                end_time=end_time,
                cpu_index=args.cpu_index,
            )

        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
