"""Error wrappers and classes common to all SoC devices."""

from __future__ import annotations

import abc
from dataclasses import dataclass

from zeus.device.exception import ZeusBaseSoCError
from zeus.device.common import deprecated_alias, DeprecatedAliasABCMeta


class ZeusSoCInitError(ZeusBaseSoCError):
    """Import error for SoC initialization failures."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


@dataclass
class SoCMeasurement(abc.ABC, metaclass=DeprecatedAliasABCMeta):
    """Represents energy consumption metrics of various subsystems on a SoC processor.

    Since subsystems available on a SoC processor are highly variable, the fields of
    this dataclass are entirely up to each derived class.

    Fields available and implemented for a specific SoC processor architecture can be
    found by referring to the SoCMeasurement derived class corresponding to that
    particular architecture (e.g., `AppleSiliconMeasurement` for Apple silicon),
    or by simply printing an instance of that derived class.

    Units: mJ
    """

    @abc.abstractmethod
    def __sub__(self, other) -> SoCMeasurement:
        """Produce a single measurement object containing differences across all fields."""
        pass

    @deprecated_alias("zeroAllFields")
    @abc.abstractmethod
    def zero_all_fields(self) -> None:
        """Set the value of all fields in the measurement object to zero."""
        pass


class SoC(abc.ABC, metaclass=DeprecatedAliasABCMeta):
    """An abstract base class for monitoring the energy consumption of a monolithic SoC processor.

    This class will be utilized by ZeusMonitor.
    """

    def __init__(self) -> None:
        """Initialize the SoC class.

        If a derived class implementation intends to rely on this base class's implementation of
        `begin_window` and `end_window`, it must invoke this constructor in its own. Otherwise, if
        it will override both of those methods, it can skip invoking this.
        """
        self.measurement_states: dict[str, SoCMeasurement] = {}

    @deprecated_alias("getAvailableMetrics")
    @abc.abstractmethod
    def get_available_metrics(self) -> set[str]:
        """Return a set of all observable metrics on the current processor."""
        pass

    @deprecated_alias("getTotalEnergyConsumption")
    @abc.abstractmethod
    def get_total_energy_consumption(self) -> SoCMeasurement:
        """Returns the total energy consumption of the SoC.

        The measurement should be cumulative; different calls to this function throughout
        the lifetime of a single `SoC` manager object should count from a fixed arbitrary
        point in time.

        Units: mJ.
        """
        pass

    @deprecated_alias("beginWindow")
    def begin_window(self, key) -> None:
        """Begin a measurement interval labeled with `key`."""
        if key in self.measurement_states:
            raise KeyError(f"Measurement window '{key}' already exists")

        self.measurement_states[key] = self.get_total_energy_consumption()

    @deprecated_alias("endWindow")
    def end_window(self, key) -> SoCMeasurement:
        """End a measurement window and return the energy consumption. Units: mJ."""
        # Retrieve the measurement taken at the start of the window.
        try:
            start_cumulative: SoCMeasurement = self.measurement_states.pop(key)
        except KeyError:
            raise KeyError(f"Measurement window '{key}' does not exist") from None

        end_cumulative: SoCMeasurement = self.get_total_energy_consumption()
        return end_cumulative - start_cumulative


class EmptySoC(SoC):
    """Empty SoC management object to be used when SoC management object is unavailable."""

    def __init__(self) -> None:
        """Initialize an empty SoC class."""
        pass

    @deprecated_alias("getAvailableMetrics")
    def get_available_metrics(self) -> set[str]:
        """Return a set of all observable metrics on the current processor."""
        return set()

    @deprecated_alias("getTotalEnergyConsumption")
    def get_total_energy_consumption(self) -> SoCMeasurement:
        """Returns the total energy consumption of the SoC.

        The measurement should be cumulative, with different calls to this function all
        counting from a fixed arbitrary point in time.

        Units: mJ.
        """
        raise ValueError("No SoC is available.")

    @deprecated_alias("beginWindow")
    def begin_window(self, key) -> None:
        """Begin a measurement interval labeled with `key`."""
        raise ValueError("No SoC is available.")

    @deprecated_alias("endWindow")
    def end_window(self, key) -> SoCMeasurement:
        """End a measurement window and return the energy consumption. Units: mJ."""
        raise ValueError("No SoC is available.")
