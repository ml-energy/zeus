"""Error wrappers and classes common to all CPU vendors."""

from __future__ import annotations

import abc
from typing import Sequence
from dataclasses import dataclass

from zeus.device.exception import ZeusBaseCPUError


@dataclass
class CpuDramMeasurement:
    """Represents a measurement of CPU and DRAM energy consumption.

    Attributes:
        cpu_mj (int): The CPU energy consumption in millijoules.
        dram_mj (Optional[int]): The DRAM energy consumption in millijoules. Defaults to None.
    """

    cpu_mj: float
    dram_mj: float | None = None

    def __sub__(self, other: CpuDramMeasurement) -> CpuDramMeasurement:
        """Subtracts the values of another CpuDramMeasurement from this one.

        Args:
            other (CpuDramMeasurement): The other CpuDramMeasurement to subtract.

        Returns:
            CpuDramMeasurement: A new CpuDramMeasurement with the result of the subtraction.
        """
        dram_mj = None
        if self.dram_mj is not None and other.dram_mj is not None:
            dram_mj = self.dram_mj - other.dram_mj
        elif self.dram_mj is not None:
            dram_mj = self.dram_mj
        elif other.dram_mj is not None:
            dram_mj = -other.dram_mj
        return CpuDramMeasurement(self.cpu_mj - other.cpu_mj, dram_mj)

    def __truediv__(self, other: int | float) -> CpuDramMeasurement:
        """Divides the values of this CpuDramMeasurement by a float.

        Args:
            other: The float to divide by.

        Returns:
            CpuDramMeasurement: A new CpuDramMeasurement with the result of the division.

        Raises:
            ZeroDivisionError: If division by zero is attempted.
        """
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero is not allowed")
            dram_mj = None
            if self.dram_mj is not None:
                dram_mj = self.dram_mj / other
            return CpuDramMeasurement(self.cpu_mj / other, dram_mj)
        else:
            return NotImplemented


class ZeusCPUInitError(ZeusBaseCPUError):
    """Import error or CPU library initialization failures."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusCPUNoPermissionError(ZeusBaseCPUError):
    """Zeus CPU exception class wrapper for No Permission to perform CPU operation."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusCPUNotFoundError(ZeusBaseCPUError):
    """Zeus CPU exception class wrapper for Not Found CPU."""

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
    def getTotalEnergyConsumption(self) -> CpuDramMeasurement:
        """Returns the total energy consumption of the specified powerzone. Units: mJ."""
        pass

    @abc.abstractmethod
    def supportsGetDramEnergyConsumption(self) -> bool:
        """Returns True if the specified CPU powerzone supports retrieving the subpackage energy consumption."""
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

    def getTotalEnergyConsumption(self, index: int) -> CpuDramMeasurement:
        """Returns the total energy consumption of the specified powerzone. Units: mJ."""
        return self.cpus[index].getTotalEnergyConsumption()

    def supportsGetDramEnergyConsumption(self, index: int) -> bool:
        """Returns True if the specified CPU powerzone supports retrieving the subpackage energy consumption."""
        return self.cpus[index].supportsGetDramEnergyConsumption()

    def __len__(self) -> int:
        """Returns the number of CPUs being tracked."""
        return len(self.cpus)


class EmptyCPUs(CPUs):
    """Empty CPUs management object to be used when CPUs management object is unavailable.

    Calls to any methods will return a value error and the length of this object will be 0
    """

    def __init__(self) -> None:
        """Instantiates empty CPUs object."""
        pass

    def __del__(self) -> None:
        """Shuts down the Intel CPU monitoring."""
        pass

    @property
    def cpus(self) -> Sequence[CPU]:
        """Returns a list of CPU objects being tracked."""
        return []

    def getTotalEnergyConsumption(self, index: int) -> CpuDramMeasurement:
        """Returns the total energy consumption of the specified powerzone. Units: mJ."""
        raise ValueError("No CPUs available.")

    def supportsGetDramEnergyConsumption(self, index: int) -> bool:
        """Returns True if the specified CPU powerzone supports retrieving the subpackage energy consumption."""
        raise ValueError("No CPUs available.")

    def __len__(self) -> int:
        """Returns 0 since the object is empty."""
        return 0
