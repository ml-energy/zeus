"""NVIDIA Jetson Nano SoC energy measurement module."""

from __future__ import annotations

import abc

from typing import Set
from pathlib import Path

import multiprocessing as mp
import time

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
        """Initialize DirectPower with the sysfs path of power."""
        self.power_path = power_path

    def measure_power(self) -> float:
        """Measure power by reading from sysfs paths."""
        power = float(self.power_path.read_text().strip())
        return power


class VoltageCurrentProduct(PowerMeasurementStrategy):
    """Computes power as product of voltage and current, read from two sysfs paths."""

    def __init__(self, voltage_path: Path, current_path: Path):
        """Initialize VoltageCurrentProduct with the sysfs paths to voltage and current."""
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
                metrics[rail_name] = {"power": str(power_path)}
            else:
                sub = {}
                if check_file(volt_path):
                    sub["volt"] = str(volt_path)
                if check_file(curr_path):
                    sub["curr"] = str(curr_path)
                if sub:
                    metrics[rail_name] = sub
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

        return metrics

    def getAvailableMetrics(self) -> Set[str]:
        """Return a set of all observable metrics on the Jetson device."""
        available_metrics = set()
        for metric in self.metric_paths:
            available_metrics.add(metric)
        return available_metrics

    def isPresent(self) -> bool:
        """Return whether we are running on a Jetson device processor."""
        return False

    def getTotalEnergyConsumption(self) -> JetsonMeasurement:
        """Returns the total energy consumption of the Jetson device.

        This measurement is cumulative. Units: mJ.
        """
        pass
