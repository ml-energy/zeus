# zeus/device/soc/jetson.py

from __future__ import annotations

from typing import List, Tuple, Set
import os
from pathlib import Path

import zeus.device.soc.common as soc_common


def check_file(path):
    return path.exists() and path.is_file()


class JetsonMeasurement(soc_common.SoCMeasurement):
    # Represents energy consumption of various subsystems on a Jetson Nano processor

    def __str__(self) -> str:
        # TODO: return a string representation of all fields and values
        pass

    def __sub__(self, other: JetsonMeasurement) -> JetsonMeasurement:
        # TODO: return a new JetsonMeasurement with subtracted field values
        pass

    def zeroAllFields(self) -> None:
        # TODO: set all internal measurement values to 0
        pass


class Jetson(soc_common.SoC):
    # An interface for obtaining the energy metrics of a Jetson Nano processor

    def __init__(self) -> None:
        super().__init__()
        self.metric_paths: List[Tuple[str, str]] = self._discover_metrics_and_paths()

    def _discover_metrics_and_paths(self):
        metrics = {}
        path = Path("/sys/bus/i2c/drivers/ina3221x")
        # subdevices = devices.glob("*")
        
        def extract_directories(path, rail_name, rail_index, type):
            if type == "label":
                power_path = path / f"power{rail_index}_input"
                volt_path = path / f"in{rail_index}_input"
                curr_path = path / f"curr{rail_index}_input"
            elif type == "rail_name":
                power_path = path / f"in_power{rail_index}_input"
                volt_path = path / f"in_voltage{rail_index}_input"
                curr_path = path / f"in_current{rail_index}_input"
            else:
                return {}

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

        for device in path.glob("*"):
            for subdevice in device.glob("*"):
                label_files = subdevice.glob("in*_label")
                rail_files = subdevice.glob("rail_name_*")

                for label_file in label_files:
                    rail_name = label_file.read_text().strip() # e.g., VDD_CPU
                    rail_index = label_file.name.split("_")[0].lstrip("in")
                    extract_directories(subdevice, rail_name, rail_index, "label")

                for rail_file in rail_files:
                    rail_name = rail_file.read_text().strip()
                    rail_index = rail_file.name.lstrip("rail_name_")
                    extract_directories(subdevice, rail_name, rail_index, "rail_name")

        return metrics

    def getAvailableMetrics(self) -> Set[str]:
        return {name for name, _ in self.metric_paths}

    def isPresent(self) -> bool:
        # TODO: check whether we're running on a Jetson device
        return False

    def getTotalEnergyConsumption(self) -> JetsonMeasurement:
        # TODO: read from /sys paths and construct JetsonMeasurement with current values
        pass
