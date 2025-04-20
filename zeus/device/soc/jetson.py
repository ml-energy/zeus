"""NVIDIA Jetson Nano SoC energy measurement module."""

from __future__ import annotations

import abc

from typing import Set
from pathlib import Path
import enum
from dataclasses import dataclass

import multiprocessing as mp
import time
import atexit
import asyncio

import zeus.device.soc.common as soc_common


def check_file(path):
    """Check if the given path exists and is a file."""
    path = Path(path)
    return path.exists() and path.is_file()


class PowerMeasurementStrategy(abc.ABC):
    """Abstract base class for two different power measurement strategies."""

    @abc.abstractmethod
    def measure_power(self) -> float:
        """Measure power in mW."""
        pass


class DirectPower(PowerMeasurementStrategy):
    """Reads power directly from a sysfs path."""

    def __init__(self, power_path: Path):
        """Initialize DirectPower paths."""
        self.power_path = power_path

    def measure_power(self) -> float:
        """Measure power by reading from sysfs paths."""
        power = float(self.power_path.read_text().strip())
        return power


class VoltageCurrentProduct(PowerMeasurementStrategy):
    """Computes power as product of voltage and current, read from two sysfs paths."""

    def __init__(self, voltage_path: Path, current_path: Path):
        """Initialize VoltageCurrentProduct paths."""
        self.voltage_path = voltage_path
        self.current_path = current_path

    def measure_power(self) -> float:
        """Measure power by reading from sysfs paths."""
        voltage = float(self.voltage_path.read_text().strip())
        current = float(self.current_path.read_text().strip())
        return (voltage * current) / 1000


class JetsonMeasurement(soc_common.SoCMeasurement):
    """Represents energy measurements for Jetson Nano subsystems."""

    def __sub__(self, other: JetsonMeasurement) -> JetsonMeasurement:
        """Return a new JetsonMeasurement with subtracted field values."""
        pass

    def zeroAllFields(self) -> None:
        """Set all internal measurement values to zero."""
        pass


class Jetson(soc_common.SoC):
    """An interface for obtaining the energy metrics of a Jetson Nano processor."""

    def __init__(self) -> None:
        """Initialize Jetson monitoring object."""
        super().__init__()
        self.metric_paths = self._discover_metrics_and_paths()
        self.power_measurement = {} # maps rail to PowerMeasurementStrategy object

        # initialize queue stuff pass into polling process, spawn polling process for first time
        # 2 commands: READ, STOP
        # send STOP command to communicate w/loop

        # Instantiate PowerMeasurementStrategy objects based on available metrics
        for rail, metrics in self.metric_paths.items():
            if "power" in metrics:
                self.power_measurement[rail] = DirectPower(metrics["power"])
            elif "volt" in metrics and "curr" in metrics:
                self.power_measurement[rail] = VoltageCurrentProduct(metrics["volt"], metrics["curr"])
            else:
                raise ValueError(
                    "Not enough measurement data to obtain power readings." # implement for which rail
                )

        # spawn polling process
        context = mp.get_context("spawn")
        self.command_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.process = context.Process(target=_polling_process_async_wrapper, args=(self.command_queue, self.result_queue, self.power_measurement))
        self.process.start()
        print("Polling process started")

        atexit.register(self._stop_process)

    def _discover_metrics_and_paths(self):
        metrics = {}
        path = Path("/sys/bus/i2c/drivers/ina3221x")

        def extract_directories(path, rail_name, rail_index, type):
            if type == "label":
                power_path = path / f"power{rail_index}_input"
                volt_path = path / f"in{rail_index}_input"
                curr_path = path / f"curr{rail_index}_input"
            else:
                power_path = path / f"in_power{rail_index}_input"
                volt_path = path / f"in_voltage{rail_index}_input"
                curr_path = path / f"in_current{rail_index}_input"

            if check_file(power_path):
                metrics[rail_name] = {"power": Path(power_path)}
            elif check_file(volt_path) and check_file(curr_path):
                sub = {}
                sub["volt"] = Path(volt_path)
                sub["curr"] = Path(curr_path)
                metrics[rail_name] = sub
            else:
                raise ValueError(
                    "Not enough measurement data to obtain power readings." # implement for which rail
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

        return metrics

    
    def getAvailableMetrics(self) -> Set[str]:
        """Return a set of all observable metrics on the Jetson device."""
        available_metrics = set()
        for metric in self.metric_paths:
            available_metrics.add(metric)
        return available_metrics

    def isPresent(self) -> bool:
        """Return whether we are running on a Jetson device processor."""
        pass

    def _stop_process(self) -> None:
        self.command_queue.put_nowait(Command.STOP)
        self.process.join()

    def getTotalEnergyConsumption(self) -> JetsonMeasurement:
        """Returns the total energy consumption of the Jetson device.

        This measurement is cumulative. Units: mJ.
        """        
        print("Sending command to command_queue")
        self.command_queue.put_nowait(Command.READ)
        print("Command sent to command_queue")
        return self.result_queue.get_nowait()


class Command(enum.Enum):
    READ = "read"
    STOP = "stop"


@dataclass
class CumulativeMeasurement:
    cpu_energy_mj: float
    gpu_energy_mj: float
    # total_energy_mj: float


def _polling_process_async_wrapper(
        command_queue: mp.Queue,
        result_queue: mp.Queue,
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
    command_queue: mp.Queue,
    result_queue: mp.Queue,
    power_measurement: dict[str, PowerMeasurementStrategy],
    poll_interval: float = 0.1,
) -> None:
    print("Polling process started 2")
    cumulative_measurement = CumulativeMeasurement(cpu_energy_mj=0.0, gpu_energy_mj=0.0)
    prev_ts = time.monotonic()

    while True:
        # TODO: the pom_in_volt naming. create a map
        print("Polling for command")
        cpu_power_mj = power_measurement["POM_5V_CPU"].measure_power()
        gpu_power_mj = power_measurement["POM_5V_GPU"].measure_power()

        current_ts = time.monotonic()

        # coros = [asyncio.to_thread(command_queue.get()), asyncio.sleep(poll_interval)]
        cmd_task = asyncio.create_task(asyncio.to_thread(command_queue.get))
        sleep_task = asyncio.create_task(asyncio.sleep(poll_interval))
        done, pending = await asyncio.wait([cmd_task, sleep_task], return_when=asyncio.FIRST_COMPLETED)
        for task in pending: 
            task.cancel()
            try:
                await task  # Properly await cancellation
            except asyncio.CancelledError:
                pass

        for task in done:
            result = task.result()
            print(f"Received command: {result}")
            if result == Command.STOP:
                break
            elif result == Command.READ:
                cpu_energy_mj = cpu_power_mj * (current_ts - prev_ts)
                gpu_energy_mj = gpu_power_mj * (current_ts - prev_ts)
                cumulative_measurement.cpu_energy_mj += cpu_energy_mj
                cumulative_measurement.gpu_energy_mj += gpu_energy_mj
                prev_ts = current_ts
                print("Sending cumulative measurement to result_queue")
                await result_queue.put(cumulative_measurement)
