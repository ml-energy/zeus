"""RAPL CPUs.

- RAPL (Running Average Power Limit):
   RAPL is a technology introduced by Intel that allows for power consumption monitoring and control at the processor and memory subsystem level. It provides mechanisms to enforce power limits and manage thermal conditions effectively.

- Power Zone:
   A power zone in the context of RAPL refers to a logical grouping of components within the CPU or system that share a common power domain. Each power zone can be monitored and controlled independently. Typical power zones include the entire package, specific cores, and memory subsystems.

- Package:
   The package refers to the physical CPU chip, which may contain multiple cores and integrated components. In RAPL, the package power domain encompasses the power consumption of all the cores and integrated units within the CPU package.

"""

from __future__ import annotations

import os
import contextlib
from typing import Sequence
import warnings
import re
from glob import glob

import zeus.device.cpu.common as cpu_common
from zeus.device.cpu.common import CpuDramMeasurement
from zeus.device.exception import ZeusBaseCPUError
from zeus.utils.logging import get_logger

logger = get_logger(name=__name__)

RAPL_DIR = "/sys/class/powercap/intel-rapl"


def rapl_is_available() -> bool:
    """Check if RAPL is available."""
    if not os.path.exists(RAPL_DIR):
        logger.info("RAPL is not supported on this CPU.")
        return False
    logger.info("RAPL is available.")
    return True


class ZeusRAPLNotSupportedError(ZeusBaseCPUError):
    """Zeus CPU exception class wrapper for RAPL not supported on CPU."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusRAPLFileInitError(ZeusBaseCPUError):
    """Zeus CPU exception class wrapper for RAPL file initialization error on CPU."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class RAPLFile:
    """RAPL File class for each RAPL file.

    This class defines the interface for interacting with a RAPL file for a package. A package can
    be a CPU or DRAM
    """

    def __init__(self, path: str) -> None:
        """Initialize RAPL file object. Each RAPL file object manages one energy_uj file."""
        self.path = path
        self.energy_uj_path = os.path.join(path, "energy_uj")
        try:
            with open(os.path.join(path, "name"), "r") as name_file:
                self.name: str = name_file.read().strip()
        except FileNotFoundError as err:
            raise ZeusRAPLFileInitError("Error reading package name") from err
        try:
            with open(self.energy_uj_path) as energy_file:
                self.last_energy = float(energy_file.read().strip())
        except FileNotFoundError as err:
            raise ZeusRAPLFileInitError("Error reading package energy") from err
        try:
            with open(
                os.path.join(path, "max_energy_range_uj"), "r"
            ) as max_energy_file:
                self.max_energy_range_uj = float(max_energy_file.read().strip())
        except FileNotFoundError as err:
            raise ZeusRAPLFileInitError(
                "Error reading package max energy range"
            ) from err

    def __str__(self) -> str:
        """Return a string representation of the RAPL file object."""
        return f"Path: {self.path}\nEnergy_uj_path: {self.energy_uj_path}\nName: {self.name}\
        \nLast_energy: {self.last_energy}\nMax_energy: {self.max_energy_range_uj}"

    def read(self) -> float:
        """Read the current energy value from the energy_uj file.

        Returns:
            int: The current energy value in millijoules.
        """
        with open(self.energy_uj_path) as energy_file:
            new_energy_uj = float(energy_file.read().strip())
        if new_energy_uj < self.last_energy:
            self.last_energy = new_energy_uj
            return (new_energy_uj + self.max_energy_range_uj) / 1000.0
        self.last_energy = new_energy_uj
        return self.last_energy / 1000.0


class RAPLCPU(cpu_common.CPU):
    """Control a single CPU that supports RAPL."""

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
        self.path = os.path.join(RAPL_DIR, f"intel-rapl:{self.cpu_index}")
        self.rapl_file: RAPLFile = RAPLFile(self.path)
        self.dram: RAPLFile | None = None
        for dir in os.listdir(self.path):
            if "intel-rapl" in dir:
                try:
                    rapl_file = RAPLFile(os.path.join(self.path, dir))
                except ZeusRAPLFileInitError as err:
                    warnings.warn(
                        f"Failed to initialize subpackage {err}", stacklevel=1
                    )
                    continue
                if rapl_file.name == "dram":
                    self.dram = rapl_file

    def getTotalEnergyConsumption(self) -> CpuDramMeasurement:
        """Returns the total energy consumption of the specified powerzone. Units: mJ."""
        cpu_mj = self.rapl_file.read()
        dram_mj = None
        if self.dram is not None:
            dram_mj = self.dram.read()
            cpu_mj -= dram_mj
        return CpuDramMeasurement(cpu_mj=cpu_mj, dram_mj=dram_mj)

    def supportsGetDramEnergyConsumption(self) -> bool:
        """Returns True if the specified CPU powerzone supports retrieving the subpackage energy consumption."""
        return self.dram is not None


class RAPLCPUs(cpu_common.CPUs):
    """RAPL CPU Manager object, containing individual RAPLCPU objects, abstracting RAPL calls and handling related exceptions."""

    def __init__(self) -> None:
        """Instantiates IntelCPUs object, setting up tracking for specified Intel CPUs."""
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
        pattern = re.compile(r"intel-rapl:(\d+)")
        for dir in sorted(glob(f"{RAPL_DIR}/intel-rapl:*")):
            parts = dir.split(':')
            if len(parts) > 1 and parts[1].isdigit():
                self._cpus.append(RAPLCPU(int(parts[1])))

    def __del__(self) -> None:
        """Shuts down the Intel CPU monitoring."""
        with contextlib.suppress(Exception):
            logger.info("Shutting down RAPL CPU monitoring.")
