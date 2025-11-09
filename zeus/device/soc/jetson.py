"""NVIDIA Jetson platform support."""

from __future__ import annotations

import abc
import asyncio
import atexit
import enum
import os
import platform
import sys
import time
import multiprocessing as mp
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from queue import Empty
from typing import TypedDict

from zeus.device.soc.common import SoC, SoCMeasurement, ZeusSoCInitError


class ZeusJetsonInitError(ZeusSoCInitError):
    """Jetson initialization failures."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


def check_file(path: Path) -> bool:
    """Check if the given path exists and is a file."""
    return path.exists() and path.is_file()


class PowerMeasurementStrategy(abc.ABC):
    """Abstract base class for two different power measurement strategies."""

    @abc.abstractmethod
    def measure_power(self) -> float:
        """Measure power in mW."""
        pass


class DirectPower(PowerMeasurementStrategy):
    """Reads power directly from a sysfs path."""

    def __init__(self, power_path: Path) -> None:
        """Initialize DirectPower paths."""
        self.power_path: Path = power_path

    def measure_power(self) -> float:
        """Measure power by reading from sysfs paths.

        Units: mW.
        """
        power: float = float(self.power_path.read_text().strip())
        return power


class VoltageCurrentProduct(PowerMeasurementStrategy):
    """Computes power as product of voltage and current, read from two sysfs paths."""

    def __init__(self, voltage_path: Path, current_path: Path) -> None:
        """Initialize VoltageCurrentProduct paths."""
        self.voltage_path: Path = voltage_path
        self.current_path: Path = current_path

    def measure_power(self) -> float:
        """Measure power by reading from sysfs paths.

        Units: mW.
        """
        voltage: float = float(self.voltage_path.read_text().strip())
        current: float = float(self.current_path.read_text().strip())
        return (voltage * current) / 1000


@dataclass
class JetsonMeasurement(SoCMeasurement):
    """Represents energy measurements for Jetson subsystems.

    All measurements are in mJ.
    """

    cpu_energy_mj: float | None = None
    gpu_energy_mj: float | None = None
    total_energy_mj: float | None = None

    def __sub__(self, other: JetsonMeasurement) -> JetsonMeasurement:
        """Produce a single measurement object containing differences across all fields."""
        if not isinstance(other, type(self)):
            raise TypeError("Subtraction is only supported between Jetson instances.")

        result = self.__class__()

        for field in fields(self):
            f_name = field.name
            value1 = getattr(self, f_name)
            value2 = getattr(other, f_name)
            if value1 is None and value2 is None:
                continue
            else:
                setattr(result, f_name, value1 - value2)

        return result

    def zero_all_fields(self) -> None:
        """Set all internal measurement values to zero."""
        for field in fields(self):
            f_name = field.name
            f_value = getattr(self, f_name)
            if isinstance(f_value, float):
                setattr(self, f_name, 0.0)
            else:
                setattr(self, f_name, None)


class DeviceMap(TypedDict, total=False):
    """Map of device names to their corresponding power measurement strategies."""

    cpu_power_mw: PowerMeasurementStrategy
    gpu_power_mw: PowerMeasurementStrategy
    total_power_mw: PowerMeasurementStrategy


class Jetson(SoC):
    """An interface for obtaining the energy metrics of a Jetson processor."""

    def __init__(self) -> None:
        """Initialize an instance of a Jetson energy monitor."""
        if not jetson_is_available():
            raise ZeusJetsonInitError("No Jetson processor was detected on the current device.")

        super().__init__()

        # Maps each power rail (cpu, gpu, and total) to a power measurement strategy
        self.power_measurement = self._discover_available_metrics()
        self.available_metrics: set[str] | None = None

        # Spawn polling process
        context = mp.get_context("spawn")
        self.command_queue = context.Queue()
        self.result_queue = context.Queue()
        self.process = context.Process(
            target=_polling_process_async_wrapper,
            args=(self.command_queue, self.result_queue, self.power_measurement),
        )
        self.process.start()
        atexit.register(self._stop_process)

    def _discover_available_metrics(self) -> DeviceMap:
        """Return available power measurement metrics per rail from the INA3221 sensor on Jetson devices.

        All official NVIDIA Jetson devices have at least 1 INA3221 power monitor that measures per-rail power usage via 3 channels.

          - https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3276/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/clock_power_setup.html#
          - https://docs.nvidia.com/jetson/archives/r35.6.1/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonXavierNxSeriesAndJetsonAgxXavierSeries.html#software-based-power-consumption-modeling
          - https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#
        """
        path = Path("/sys/bus/i2c/drivers/ina3221x")

        metric_paths: dict[str, dict[str, Path]] = {}
        power_measurement: DeviceMap = {}

        def extract_directories(path: Path, rail_name: str, rail_index: str, type: str) -> None:
            """Extract file paths for power, voltage, and current measurements based on the rail naming type."""
            rail_name_lower = rail_name.lower()

            if "cpu" in rail_name_lower:
                rail_name_simplified = "cpu_power_mw"
            elif "gpu" in rail_name_lower:
                rail_name_simplified = "gpu_power_mw"
            elif "system" in rail_name_lower or "_in" in rail_name_lower or "total" in rail_name_lower:
                rail_name_simplified = "total_power_mw"
            else:
                return  # Skip unsupported rail types

            if type == "label":
                power_path = path / f"power{rail_index}_input"
                volt_path = path / f"in{rail_index}_input"
                curr_path = path / f"curr{rail_index}_input"
            else:
                power_path = path / f"in_power{rail_index}_input"
                volt_path = path / f"in_voltage{rail_index}_input"
                curr_path = path / f"in_current{rail_index}_input"

            if check_file(power_path):
                metric_paths[rail_name_simplified] = {"power": Path(power_path)}
            elif check_file(volt_path) and check_file(curr_path):
                metric_paths[rail_name_simplified] = {
                    "volt": Path(volt_path),
                    "curr": Path(curr_path),
                }
            # Else, skip the rail due to insufficient metrics for power

        for device in path.glob("*"):
            for subdevice in device.glob("*"):
                # Get the files containing rail names.
                label_files = subdevice.glob("in*_label")
                rail_files = subdevice.glob("rail_name_*")
                # For each rail name, get its respective power, voltage, current paths.
                for label_file in label_files:
                    rail_name = label_file.read_text().strip()
                    rail_index = label_file.name.split("_")[0].lstrip("in")
                    extract_directories(subdevice, rail_name, rail_index, "label")
                for rail_file in rail_files:
                    rail_name = rail_file.read_text().strip()
                    rail_index = rail_file.name.split("rail_name_", 1)[-1]
                    extract_directories(subdevice, rail_name, rail_index, "rail_name")

        # Instantiate PowerMeasurementStrategy objects based on available metrics
        for rail, metrics in metric_paths.items():
            if "power" in metrics:
                power_measurement[rail] = DirectPower(metrics["power"])
            elif "volt" in metrics and "curr" in metrics:
                power_measurement[rail] = VoltageCurrentProduct(metrics["volt"], metrics["curr"])
            # Else, skip the rail due to insufficient metrics for power
        return power_measurement

    def get_available_metrics(self) -> set[str]:
        """Return a set of all observable metrics on the Jetson device."""
        if self.available_metrics is None:
            result: JetsonMeasurement = self.get_total_energy_consumption()
            available_metrics = set()

            metrics_dict = asdict(result)
            for f_name, f_value in metrics_dict.items():
                if f_value is not None:
                    available_metrics.add(f_name)

            self.available_metrics = available_metrics
        return self.available_metrics

    def _stop_process(self) -> None:
        """Kill the polling process."""
        self.command_queue.put_nowait(Command.STOP)
        self.process.join(timeout=1.0)
        self.process.kill()

    def get_total_energy_consumption(self, timeout: float = 15.0) -> JetsonMeasurement:
        """Returns the total energy consumption of the Jetson device. This measurement is cumulative.

        Units: mJ.
        """
        self.command_queue.put(Command.READ)
        return self.result_queue.get(timeout=timeout)


class Command(enum.Enum):
    """Provide commands for the polling process."""

    READ = "read"
    STOP = "stop"


def _polling_process_async_wrapper(
    command_queue: mp.Queue[Command],
    result_queue: mp.Queue[JetsonMeasurement],
    power_measurement: DeviceMap,
) -> None:
    """Function wrapper for the asynchronous energy polling process."""
    asyncio.run(
        _polling_process_async(
            command_queue,
            result_queue,
            power_measurement,
        )
    )


async def _polling_process_async(
    command_queue: mp.Queue[Command],
    result_queue: mp.Queue[JetsonMeasurement],
    power_measurement: DeviceMap,
) -> None:
    """Continuously polls for accumulated energy measurements for CPU, GPU, and total power, listening for commands to stop or return the measurement."""
    cumulative_measurement = JetsonMeasurement(
        cpu_energy_mj=0.0 if "cpu_power_mw" in power_measurement else None,
        gpu_energy_mj=0.0 if "gpu_power_mw" in power_measurement else None,
        total_energy_mj=0.0 if "total_power_mw" in power_measurement else None,
    )

    prev_ts = time.monotonic()

    while True:
        current_ts: float = time.monotonic()
        dt: float = current_ts - prev_ts

        if "cpu_power_mw" in power_measurement:
            cpu_power_mw = power_measurement["cpu_power_mw"].measure_power()
            cpu_energy_mj = cpu_power_mw * dt
            cumulative_measurement.cpu_energy_mj = (cumulative_measurement.cpu_energy_mj or 0.0) + cpu_energy_mj
        if "gpu_power_mw" in power_measurement:
            gpu_power_mw = power_measurement["gpu_power_mw"].measure_power()
            gpu_energy_mj = gpu_power_mw * dt
            cumulative_measurement.gpu_energy_mj = (cumulative_measurement.gpu_energy_mj or 0.0) + gpu_energy_mj
        if "total_power_mw" in power_measurement:
            total_power_mw = power_measurement["total_power_mw"].measure_power()
            total_energy_mj = total_power_mw * dt
            cumulative_measurement.total_energy_mj = (cumulative_measurement.total_energy_mj or 0.0) + total_energy_mj

        prev_ts = current_ts

        try:
            command = await asyncio.to_thread(
                command_queue.get,
                timeout=0.1,
            )
        except Empty:
            # Update energy and do nothing
            continue

        if command == Command.STOP:
            break
        if command == Command.READ:
            # Update and return energy measurement
            result_queue.put(cumulative_measurement)


def jetson_is_available() -> bool:
    """Return if the current processor is a Jetson device."""
    if sys.platform != "linux" or platform.processor() != "aarch64":
        return False

    return os.path.exists("/usr/lib/aarch64-linux-gnu/tegra") or os.path.exists("/etc/nv_tegra_release")
