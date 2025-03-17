"""Error wrappers and classes common to all SoC devices."""

from __future__ import annotations

import abc
from dataclasses import dataclass

from zeus.device.exception import ZeusBaseSoCError


class ZeusSoCInitError(ZeusBaseSoCError):
    """Import error for SoC initialization failures."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


@dataclass
class SoCMeasurement(abc.ABC):
    """Represents energy consumption metrics of various subsystems on a SoC processor.

    Since subsystems available on a SoC processor are highly variable, the fields of
    this dataclass are entirely up to each derived class.

    Fields available and implemented for a specific SoC processor architecture can be
    found by referring to the SoCMeasurement derived class corresponding to that
    particular architecture (e.g., `AppleSiliconMeasurement` for Apple silicon),
    or by simply printing an instance of that derived class.
    """
    
    @abc.abstractmethod
    def __str__(self):
        """Show all fields and their observed values in the measurement object."""
        pass


class SoC(abc.ABC):
    """An abstract base class for monitoring the energy consumption of a monolithic SoC processor.
    
    This class will be utilized by ZeusMonitor.
    """

    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def getAvailableMetrics(self) -> Set[str]:
        """Return a set of all observable metrics on the current processor."""
        pass

    @abc.abstractmethod
    def getTotalEnergyConsumption(self, index: int) -> SoCMeasurement:
        """Returns the total energy consumption of the SoC. Units: mJ."""
    
    def beginWindow(self, key) -> None:
        """Begin a measurement interval labeled with `key`."""
        pass
    
    def endWindow(self, key) -> SoCMeasurement:
        """End the measurement interval labeled with `key` and return the energy
        consumed by processor subsystems during the interval. Units: mJ."""
        pass
