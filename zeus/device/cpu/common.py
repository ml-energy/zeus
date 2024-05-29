"""Error wrappers and classes common to all CPU vendors."""

from __future__ import annotations

import abc
from typing import Sequence

from zeus.device.exception import ZeusBaseCPUError


class ZeusCPUInitError(ZeusBaseCPUError):
    """Import error or CPU library initialization failures."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusCPUNoPermissionError(ZeusBaseCPUError):
    """Zeus CPU exception class Wrapper for No Permission to perform CPU operation."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusCPUNotFoundError(ZeusBaseCPUError):
    """Zeus CPU exception class Wrapper for Not Found CPU."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class CPU(abc.ABC):
    """Abstract base class for CPU management.

    This class defines the interface for interacting with CPUs, subclasses should implement the methods to interact with specific CPU libraries.
    """

    def __init__(self, cpu_index: int) -> None:
        """Initialize the CPU with a specified index."""
        self.cpu_index = cpu_index

    @abc.abstractmethod
    def getTotalEnergyConsumption(self) -> int:
        """Returns the total energy consumption of the specified powerzone. Units: mJ."""
        pass

    @abc.abstractmethod
    def supportsGetSubpackageEnergyConsumption(self) -> bool:
        """Returns True if the specified CPU powerzone supports retrieving the subpackage energy consumption."""
        pass

    @abc.abstractmethod
    def getSubpackageEnergyConsumption(self) -> int:
        """Returns the energy consumption of the subpackages inside the specified powerzone. Units: mJ."""
        pass


class CPUs(abc.ABC):
    """An abstract base class for CPU manager object.

    This class defines the essential interface and common functionality for CPU management, instantiating multiple `CPU` objects for each CPU being tracked.
    Forwards the call for a specific method to the corresponding CPU object.
    """

    @abc.abstractmethod
    def __init__(self) -> None:
        """Initializes the CPU management library to communicate with the CPU driver and sets up tracking for specified CPUs."""
        pass

    @abc.abstractmethod
    def __del__(self) -> None:
        """Shuts down the CPU monitoring library to release resources and clean up."""
        pass

    @property
    @abc.abstractmethod
    def cpus(self) -> Sequence[CPU]:
        """Returns a list of CPU objects being tracked."""
        pass

    def getTotalEnergyConsumption(self, index: int) -> int:
        """Returns the total energy consumption of the specified powerzone. Units: mJ."""
        return self.cpus[index].getTotalEnergyConsumption()

    def supportsGetSubpackageEnergyConsumption(self, index: int) -> bool:
        """Returns True if the specified CPU powerzone supports retrieving the subpackage energy consumption."""
        return self.cpus[index].supportsGetSubpackageEnergyConsumption()

    def getSubpackageEnergyConsumption(self, index: int) -> int:
        """Returns the energy consumption of the subpackages inside the specified powerzone. Units: mJ."""
        return self.cpus[index].getSubpackageEnergyConsumption()

    def __len__(self) -> int:
        """Returns the number of CPUs being tracked."""
        return len(self.cpus)

