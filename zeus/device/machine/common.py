"""Error wrappers and classes common to all whole-machine power meters.

A machine here is the box rather than the chip. The meter is off-die: a PMIC
metering board rails, a BMC answering over IPMI, a wall meter. That is the
difference from `zeus.device.soc`, whose sensors are on the processor itself and
so can only ever see part of the draw.

Every implementation must be able to report whole-machine energy, because that
is the one figure any of these sources can give. Anything finer is up to the
derived measurement class, since a PMIC that breaks out a DRAM rail and a BMC
that reports one number for the chassis have very little in common below that.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, fields
from typing import TypeVar

from zeus.device.exception import ZeusBaseMachineError

# So that subtracting two measurements of a concrete type gives that type back
# rather than the base, which a caller reaching for a platform-specific field needs.
TMachineMeasurement = TypeVar("TMachineMeasurement", bound="MachineMeasurement")


class ZeusMachineInitError(ZeusBaseMachineError):
    """Import error or machine metering initialization failures."""

    def __init__(self, message: str) -> None:
        """Initialize the exception object."""
        super().__init__(message)


@dataclass
class MachineMeasurement(abc.ABC):
    """Energy consumed by a whole machine, and by whichever domains it can separate.

    `machine_energy_mj` is the only field guaranteed to be present. Derived
    classes add the domains their hardware can actually distinguish, for example
    a CPU rail and a DRAM rail on a board whose PMIC meters them separately.
    Refer to the derived class for a specific platform, or print an instance.

    Units: mJ
    """

    machine_energy_mj: float

    def __sub__(self: TMachineMeasurement, other: TMachineMeasurement) -> TMachineMeasurement:
        """Return a measurement holding the difference across every field.

        Fields that are None on either side stay None, because a domain that
        was unreadable for part of a window has no meaningful delta.
        """
        if type(self) is not type(other):
            raise TypeError(f"cannot subtract {type(other).__name__} from {type(self).__name__}")
        diff = {}
        for f in fields(self):
            mine, theirs = getattr(self, f.name), getattr(other, f.name)
            diff[f.name] = None if mine is None or theirs is None else mine - theirs
        return type(self)(**diff)

    def zero_all_fields(self) -> None:
        """Set every field that carries a value to zero, leaving None fields alone."""
        for f in fields(self):
            if getattr(self, f.name) is not None:
                setattr(self, f.name, 0.0)


class Machine(abc.ABC):
    """Abstract base class for metering the energy of a whole machine.

    Used by ZeusMonitor in the same way as the SoC and CPU managers.
    """

    @abc.abstractmethod
    def get_available_metrics(self) -> set[str]:
        """Return the set of measurement fields this machine actually populates."""

    @abc.abstractmethod
    def get_total_energy_consumption(self) -> MachineMeasurement:
        """Return cumulative energy since a fixed arbitrary point.

        Successive calls over the lifetime of one manager object count from the
        same origin, so a caller can difference any two of them.

        Units: mJ.
        """

    @abc.abstractmethod
    def begin_window(self, key: str, restart: bool = False) -> None:
        """Begin a measurement interval labeled with `key`.

        Args:
            key: Unique name of the measurement window.
            restart: If True and the window already exists, cancel the existing
                window and start a new one.
        """

    @abc.abstractmethod
    def end_window(self, key: str) -> MachineMeasurement:
        """End a measurement window and return the energy consumed. Units: mJ."""


class EmptyMachine(Machine):
    """Stand-in used when no machine-level meter is available."""

    def get_available_metrics(self) -> set[str]:
        """Return an empty set, since nothing is measurable."""
        return set()

    def get_total_energy_consumption(self) -> MachineMeasurement:
        """Raise, since nothing is measurable."""
        raise ValueError("No machine power meter is available.")

    def begin_window(self, key: str, restart: bool = False) -> None:
        """Raise, since nothing is measurable."""
        raise ValueError("No machine power meter is available.")

    def end_window(self, key: str) -> MachineMeasurement:
        """Raise, since nothing is measurable."""
        raise ValueError("No machine power meter is available.")
