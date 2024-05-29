"""Intel CPUs."""

from __future__ import annotations

import functools
import os
import contextlib
from typing import Sequence, Dict, List

import zeus.device.cpu.common as cpu_common
from zeus.device.exception import ZeusBaseCPUError
from zeus.utils.logging import get_logger

logger = get_logger(name=__name__)

DIR: str = "/sys/class/powercap/intel-rapl"

def rapl_is_available() -> bool:
    """Check if RAPL is available."""
    if not os.path.exists(DIR):
        logger.info("RAPL is not supported on this CPU.")
        return False
    logger.info("RAPL is available.")
    return True

class ZeusRAPLNotSupportedError(ZeusBaseCPUError):
    """Zeus CPU exception class Wrapper for RAPL not supported on CPU."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)

class ZeusRAPLFileInitError(ZeusBaseCPUError):
    """Zeus CPU exception class Wrapper for RAPL file initialization error on CPU."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)

class RAPLFile:
    """RAPL File class for each RAPL file.

    This class defines the interface for interacting with a RAPL file for a package. A package can
    be a CPU or DRAM
    """

    def __init__(self, path: str) -> None:
        self.path: str = path
        self.energy_uj_path: str = os.path.join(path, "energy_uj")
        try:
            with open(os.path.join(path, "name"), 'r') as name_file:
                self.name: str = name_file.read().strip()
        except:
            raise ZeusRAPLFileInitError(
                "Error reading package name"
            )
        try:
            with open(self.energy_uj_path) as energy_file:
                self.last_energy: int = int(float(energy_file.read().strip())/10**3)
        except:
            raise ZeusRAPLFileInitError(
                "Error reading package energy"
            )
        try:
            with open(os.path.join(path, "max_energy_range_uj"), 'r') as max_energy_file:
                self.max_energy_range_uj: int = int(float(max_energy_file.read().strip())/10**3)
        except:
            raise ZeusRAPLFileInitError(
                "Error reading package max energy range"
            )

    def __str__(self) -> str:
        return f"Path: {self.path}\nEnergy_uj_path: {self.energy_uj_path}\nName: {self.name}\
        \nLast_energy: {self.last_energy}\nMax_energy: {self.max_energy_range_uj}"

    def read(self) -> int:
        with open(self.energy_uj_path) as energy_file:
            self.last_energy = int(float(energy_file.read().strip())/10**3)
        return self.last_energy

    def read_delta(self) -> int:
        last_energy: int = self.last_energy
        new_energy: int = self.read()
        if new_energy < last_energy:
            return new_energy + self.max_energy_range_uj - last_energy
        return new_energy - last_energy

class INTELCPU(cpu_common.CPU):
    """Control a Single Intel CPU using RAPL interface."""

    def __init__(self, cpu_index: int) -> None:
        """Initialize the Intel CPU with a specified index."""
        super().__init__(cpu_index)
        self._get_powerzone()

    _exception_map = {
        FileNotFoundError: cpu_common.ZeusCPUNotFoundError,
        PermissionError: cpu_common.ZeusCPUNoPermissionError,
        OSError: cpu_common.ZeusCPUInitError,
    }

    def _get_powerzone(self) -> None:
        self.path = os.path.join(DIR, f"intel-rapl:{self.cpu_index}")
        self.rapl_file: RAPLFile = RAPLFile(self.path)
        self.subpackages: List = []
        for dir in os.listdir(self.path):
            if "intel-rapl" in dir:
                self.subpackages.append(RAPLFile(os.path.join(self.path, dir)))

    def getTotalEnergyConsumption(self) -> int:
        """Returns the total energy consumption of the specified powerzone. Units: mJ."""
        return self.rapl_file.read()

    def supportsGetSubpackageEnergyConsumption(self) -> bool:
        """Returns True if the specified CPU powerzone supports retrieving the subpackage energy consumption."""
        return len(self.subpackages)!=0

    def getSubpackageEnergyConsumption(self) -> int:
        """Returns the energy consumption of the subpackages inside the specified powerzone. Units: mJ."""
        raise NotImplementedError("Subpackage readings not implemented")
        return 0


class INTELCPUs(cpu_common.CPUs):
    """Intel CPU Manager object, containing individual IntelCPU objects, abstracting RAPL calls and handling related exceptions."""

    def __init__(self) -> None:
        """Instantiates IntelCPUs object, setting up tracking for specified Intel CPUs.
        """
        if not rapl_is_available():
            raise ZeusRAPLNotSupportedError("RAPL is not supported on this CPU.")
        self._init_cpus()

    @property
    def cpus(self) -> Sequence[cpu_common.CPU]:
        """Returns a list of CPU objects being tracked."""
        return self._cpus

    def _init_cpus(self) -> None:
        """Initialize all Intel CPUs."""
        self._cpus = []
        for index in range(2):
            try:
                self._cpus.append(INTELCPU(index))
            except:
                continue

    def __del__(self) -> None:
        """Shuts down the Intel CPU monitoring."""
        with contextlib.suppress(Exception):
            logger.info("Shutting down Intel CPU monitoring.")
