"""NVIDIA Jetson Nano SoC energy measurement module."""

from __future__ import annotations

import abc
import asyncio
import atexit
import enum
import time
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import zeus.device.soc.common as soc_common


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

    def __init__(self, power_path: Path):  ###
        """Initialize DirectPower paths."""
        self.power_path = power_path

    def measure_power(self) -> float:
        """Measure power by reading from sysfs paths.

        Units: mW.
        """
        power = float(self.power_path.read_text().strip())
        return power / 1000


class VoltageCurrentProduct(PowerMeasurementStrategy):
    """Computes power as product of voltage and current, read from two sysfs paths."""

    def __init__(self, voltage_path: Path, current_path: Path):  ###
        """Initialize VoltageCurrentProduct paths."""
        self.voltage_path = voltage_path
        self.current_path = current_path

    def measure_power(self) -> float:
        """Measure power by reading from sysfs paths.

        Units: mW.
        """
        voltage = float(self.voltage_path.read_text().strip())
        current = float(self.current_path.read_text().strip())
        return (voltage * current) / 1000


@dataclass
class JetsonMeasurement(soc_common.SoCMeasurement):
    """Represents energy measurements for Jetson Nano subsystems."""

    cpu_energy_mj: float = 0.0
    gpu_energy_mj: float = 0.0
    total_energy_mj: float = 0.0

    def __sub__(self, other: JetsonMeasurement) -> JetsonMeasurement:
        """Return a new JetsonMeasurement with subtracted field values."""
        if not isinstance(other, JetsonMeasurement):
            raise ValueError("Other must be a JetsonMeasurement object.")

        return JetsonMeasurement(
            cpu_energy_mj=self.cpu_energy_mj - other.cpu_energy_mj,
            gpu_energy_mj=self.gpu_energy_mj - other.gpu_energy_mj,
            total_energy_mj=self.total_energy_mj - other.total_energy_mj,
        )

    def zeroAllFields(self) -> None:
        """Set all internal measurement values to zero."""
        self.cpu_energy_mj = 0.0
        self.gpu_energy_mj = 0.0
        self.total_energy_mj = 0.0


class Jetson(soc_common.SoC):
    """An interface for obtaining the energy metrics of a Jetson Nano processor."""

    def __init__(self) -> None:
        """Initialize Jetson monitoring object."""
        super().__init__()
        ###
        self.power_measurement = self._discover_available_metrics()

        # spawn polling process
        context = mp.get_context("fork")
        self.command_queue = context.Queue()
        self.result_queue = context.Queue()
        self.process = context.Process(
            target=_polling_process_async_wrapper,
            args=(self.command_queue, self.result_queue, self.power_measurement),
        )
        self.process.start()

        self.command_queue.put("read")
        response = self.result_queue.get()
        print(response)

        atexit.register(self._stop_process)

    def _discover_available_metrics(self) -> dict[str, PowerMeasurementStrategy]:
        metric_paths = {}
        power_measurement = {}

        # todo: cite documentation for ina3221
        path = Path("/sys/bus/i2c/drivers/ina3221x")

        def extract_directories(path, rail_name, rail_index, type):  ###
            rail_name_lower = rail_name.lower()

            if "cpu" in rail_name_lower:
                rail_name_simplified = "cpu"
            elif "gpu" in rail_name_lower:
                rail_name_simplified = "gpu"
            elif (
                "system" in rail_name_lower
                or "_in" in rail_name_lower
                or "total" in rail_name_lower
            ):
                rail_name_simplified = "total"
            else:
                return  # skip unsupported rail types

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
                sub = {}
                sub["volt"] = Path(volt_path)
                sub["curr"] = Path(curr_path)
                metric_paths[rail_name_simplified] = sub
            else:
                raise ValueError(
                    "Not enough measurement data to obtain power readings."
                )

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
                power_measurement[rail] = VoltageCurrentProduct(
                    metrics["volt"], metrics["curr"]
                )
            else:
                raise ValueError(
                    f"Not enough measurement data to obtain power readings for '{rail}'."
                )

        return power_measurement

    def getAvailableMetrics(self) -> set[str]:
        """Return a set of all observable metrics on the Jetson device."""
        return {rail + "_mj" for rail in self.power_measurement}

    def _stop_process(self) -> None:
        """Kill the polling process."""
        self.command_queue.put_nowait("read")
        self.process.join()

    def getTotalEnergyConsumption(self) -> JetsonMeasurement:
        """Returns the total energy consumption of the Jetson device. This measurement is cumulative.

        Units: mJ.
        """
        self.command_queue.put_nowait(Command.READ)
        print("Command sent to command_queue")
        return self.result_queue.get(timeout=15)


class Command(enum.Enum):
    """Provide commands for the polling process."""

    READ = "read"
    STOP = "stop"


def _polling_process_async_wrapper(
    command_queue: mp.Queue[Command],
    result_queue: mp.Queue[JetsonMeasurement],
    power_measurement: dict[str, PowerMeasurementStrategy],
    poll_interval: float = 0.1,
) -> None:
    asyncio.run(
        _polling_process_async(
            command_queue,
            result_queue,
            power_measurement,
            poll_interval,
        )
    )


async def _polling_process_async(
    command_queue: mp.Queue[Command],
    result_queue: mp.Queue[JetsonMeasurement],
    power_measurement: dict[str, PowerMeasurementStrategy],
    poll_interval: float = 0.1,
) -> None:
    print("Polling process started")
    cumulative_measurement = JetsonMeasurement(
        cpu_energy_mj=0.0, gpu_energy_mj=0.0, total_energy_mj=0.0
    )

    prev_ts = time.monotonic()

    while True:
        print("Polling for command")
        cpu_power_mw = power_measurement["cpu"].measure_power()
        gpu_power_mw = power_measurement["gpu"].measure_power()
        total_power_mw = power_measurement["total"].measure_power()

        current_ts = time.monotonic()
        dt = current_ts - prev_ts

        cpu_energy_mj = cpu_power_mw * dt
        gpu_energy_mj = gpu_power_mw * dt
        total_energy_mj = total_power_mw * dt

        cumulative_measurement.cpu_energy_mj += cpu_energy_mj
        cumulative_measurement.gpu_energy_mj += gpu_energy_mj
        cumulative_measurement.total_energy_mj += total_energy_mj

        prev_ts = current_ts

        try:
            command = await asyncio.wait_for(
                asyncio.to_thread(command_queue.get),
                timeout=0.1,
            )
        except asyncio.TimeoutError:
            # Update energy and do nothing
            continue

        if command == "stop":
            print("Bye!")
            break
        if command == "read":
            # Update energy
            print("Sending cumulative measurement to result_queue")
            result_queue.put(cumulative_measurement)
