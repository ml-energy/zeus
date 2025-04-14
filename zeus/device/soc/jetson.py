# zeus/device/soc/jetson.py

from __future__ import annotations

from typing import List, Tuple, Set
# from .common import cat, check_file
import os
import logging

import zeus.device.soc.common as soc_common

logger = logging.getLogger(__name__)

def cat(path):
    with open(path, 'r') as f:
        return f.readline().rstrip('\x00')

def check_file(path):
    return os.path.isfile(path) and os.access(path, os.R_OK)

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

    def find_driver_power_folders(self, path):
        subdirectories = []
        sensors = os.listdir(path)
        for item in sensors:
            if os.path.isdir(os.path.join(path, item)):
                sensor_path = "{base_path}/{item}".format(base_path=path, item=item)
                if 'hwmon' in item:
                    # Go 1 level deeper & get first subdirectory (which contains actual readings)
                    hwmon_name = os.listdir(sensor_path)[0] # hwmon0/
                    sensor_path = "{base_path}/{item}".format(base_path=sensor_path, item=hwmon_name) # /sys/.../hwmon/hwmon0/
                    subdirectories += [sensor_path]
                elif 'iio:device' in item:
                    subdirectories += [sensor_path] # /sys/.../iio:device0/
        return subdirectories
    
    def locate_directories(self, path):
        sensor_paths = {}
        # loops through each sensor, looking for ones ending in _label or starting w/rail_name_
        for item in os.listdir(path):
            power_label_path = "{path}/{item}".format(path=path, item=item)
            if item.endswith("_label"):
                # raw_name = cat(power_label_path).strip()
                label = cat(power_label_path).strip().lower() # sensor name (e.g. "VDD_CPU")
                number_port = item.split("_")[0].strip("in")
                volt_path = os.path.join(path, f"in{number_port}_input")
                curr_path = os.path.join(path, f"curr{number_port}_input")
                power_path = os.path.join(path, f"power{number_port}_input")
            elif item.startswith("rail_name_"):
                # raw_name = cat(power_label_path).strip()
                label = cat(power_label_path).strip().lower()
                number_port = item.lstrip("rail_name_")
                volt_path = os.path.join(path, f"in_voltage{number_port}_input")
                curr_path = os.path.join(path, f"in_current{number_port}_input")
                power_path = os.path.join(path, f"in_power{number_port}_input")
            else:
                continue

            # if check_file(power_path):
            #     sensor_paths[f"{label}_power"] = power_path
            # else:
            #     if check_file(volt_path):
            #         sensor_paths[f"{label}_volt"] = volt_path
            #     if check_file(curr_path):
            #         sensor_paths[f"{label}_curr"] = curr_path

            if check_file(power_path):
                sensor_paths[label] = {"power": power_path}
            else:
                sub = {}
                if check_file(volt_path):
                    sub["volt"] = volt_path
                if check_file(curr_path):
                    sub["curr"] = curr_path
                if sub:
                    sensor_paths[label] = sub

        return sensor_paths

    def _discover_metrics_and_paths(self):
        """Replicates jetson-stats' /sys path & metric discovery logic."""
        metrics = {}
        i2c_path = "/sys/bus/i2c/devices"

        if not os.path.isdir(i2c_path):
            logger.error("Folder {root_dir} doesn't exist".format(root_dir=i2c_path))
            return metrics
        power_i2c_sensors = {}
        devices = os.listdir(i2c_path) # lists all items in i2c_path directory (i2c-0, 1-0040, etc.)

        for device in devices:
            device_path = "{base_path}/{item}".format(base_path=i2c_path, item=device)
            name_path = "{path}/name".format(path=device_path)
            if os.path.isfile(name_path): 
                raw_name = cat(name_path).strip()
                if 'ina3221' not in raw_name:
                    # power_i2c_sensors[device] = self.find_driver_power_folders(device_path)
                    continue
                power_i2c_sensors[device] = self.find_driver_power_folders(device_path)
        
        for name, paths in power_i2c_sensors.items():
            for path in paths: # i.e. /sys/bus/i2c/devices/1-0040/iio:device0
                sensors = self.locate_directories(path)
                metrics.update(sensors)

        return metrics
    


    def getAvailableMetrics(self) -> Set[str]:
        return {name for name, _ in self.metric_paths}

    def isPresent(self) -> bool:
        # TODO: check whether we're running on a Jetson device
        return False

    def getTotalEnergyConsumption(self) -> JetsonMeasurement:
        # TODO: read from /sys paths and construct JetsonMeasurement with current values
        pass
