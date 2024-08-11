"""RAPL CPUs.

- RAPL (Running Average Power Limit):
   RAPL is a technology introduced by Intel that allows for power consumption monitoring and control at the processor and memory subsystem level. It provides mechanisms to enforce power limits and manage thermal conditions effectively.

- Power Zone:
   A power zone in the context of RAPL refers to a logical grouping of components within the CPU or system that share a common power domain. Each power zone can be monitored and controlled independently. Typical power zones include the entire package, specific cores, and memory subsystems.

- Package:
   The package refers to the physical CPU chip, which may contain multiple cores and integrated components. In RAPL, the package power domain encompasses the power consumption of all the cores and integrated units within the CPU package.

"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import time
import warnings
from functools import lru_cache
from glob import glob
from multiprocessing.sharedctypes import Synchronized
from typing import Sequence

import zeus.device.cpu.common as cpu_common
from zeus.device.cpu.common import CpuDramMeasurement
from zeus.device.exception import ZeusBaseCPUError
from zeus.utils.logging import get_logger

logger = get_logger(name=__name__)

RAPL_DIR = "/sys/class/powercap/intel-rapl"

# Assuming a maximum power draw of 1000 Watts when we are polling every 0.1 seconds, the maximum
# amount the RAPL counter would increase
RAPL_COUNTER_MAX_INCREASE = 1000 * 1e6 * 0.1


class RaplWraparoundTracker:
    """Monitor the wrapping around of RAPL counters.

    This class acts as a lower level wrapper around a Python process that polls
    the wrapping of RAPL counters. This is primarily used by
    [`RAPLCPUs`][zeus.device.cpu.rapl.RAPLCPUs].

    !!! Warning
        Since the monitor spawns a child process, **it should not be instantiated as a global variable**.
        Python puts a protection to prevent creating a process in global scope.
        Refer to the "Safe importing of main module" section in the
        [Python documentation](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods)
        for more details.

    Attributes:
        rapl_file_path (str): File path of rapl file to track wraparounds for.
        max_energy_uj (float): Max value of rapl counter for `rapl_file_path` file. Used to
            determine the sleep period between polls
    """

    def __init__(
        self,
        rapl_file_path: str,
        max_energy_uj: float,
    ) -> None:
        """Initialize the rapl monitor.

        Args:
            rapl_file_path: File path where the RAPL file is located
            max_energy_uj: Max energy range uj value
        """
        if not os.path.exists(rapl_file_path):
            raise ValueError(f"{rapl_file_path} is not a valid file path")

        # Set up logging.
        self.logger = get_logger(type(self).__name__)

        self.logger.info("Monitoring wrap around of %s", rapl_file_path)

        context = mp.get_context("spawn")
        self.wraparound_counter = context.Value("i", 0)
        # Spawn the power polling process.
        atexit.register(self._stop)
        self.process = context.Process(
            target=_polling_process,
            args=(rapl_file_path, max_energy_uj, self.wraparound_counter),
        )
        self.process.start()

    def _stop(self) -> None:
        """Stop monitoring power usage."""
        if self.process is not None:
            self.process.terminate()
            self.process.join(timeout=1.0)
            self.process.kill()
            self.process = None

    def get_num_wraparounds(self) -> int:
        """Get the number of wraparounds detected by the polling process."""
        with self.wraparound_counter.get_lock():
            return self.wraparound_counter.value


def _polling_process(
    rapl_file_path: str, max_energy_uj: float, wraparound_counter: Synchronized[int]
) -> None:
    """Check for wraparounds in the specified rapl file."""
    try:
        with open(rapl_file_path) as rapl_file:
            last_energy_uj = float(rapl_file.read().strip())
        while True:
            sleep_time = 1.0
            with open(rapl_file_path, "r") as rapl_file:
                energy_uj = float(rapl_file.read().strip())
            if max_energy_uj - energy_uj < RAPL_COUNTER_MAX_INCREASE:
                sleep_time = 0.1
            if energy_uj < last_energy_uj:
                with wraparound_counter.get_lock():
                    wraparound_counter.value += 1
            last_energy_uj = energy_uj
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        return


@lru_cache(maxsize=1)
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

        self.wraparound_tracker = RaplWraparoundTracker(
            self.energy_uj_path, self.max_energy_range_uj
        )

    def __str__(self) -> str:
        """Return a string representation of the RAPL file object."""
        return f"RAPLFile(Path: {self.path}\nEnergy_uj_path: {self.energy_uj_path}\nName: {self.name}\
        \nLast_energy: {self.last_energy}\nMax_energy: {self.max_energy_range_uj})"

    def read(self) -> float:
        """Read the current energy value from the energy_uj file.

        Returns:
            The current energy value in millijoules.
        """
        with open(self.energy_uj_path) as energy_file:
            new_energy_uj = float(energy_file.read().strip())
        num_wraparounds = self.wraparound_tracker.get_num_wraparounds()
        return (new_energy_uj + num_wraparounds * self.max_energy_range_uj) / 1000.0


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
        for dir in sorted(glob(f"{RAPL_DIR}/intel-rapl:*")):
            parts = dir.split(":")
            if len(parts) > 1 and parts[1].isdigit():
                self._cpus.append(RAPLCPU(int(parts[1])))

    def __del__(self) -> None:
        """Shuts down the Intel CPU monitoring."""
        pass
