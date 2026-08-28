"""Abstraction layer for whole-machine power meters.

The main function of this module is [`get_machine`][zeus.device.machine.get_machine],
which returns a machine manager object specific to the platform.

A machine meter sits off the die: a PMIC metering board rails, a BMC answering
over IPMI, a wall meter. That is the distinction from
[`zeus.device.soc`][zeus.device.soc], whose sensors live on the processor and so
can only ever account for part of the machine's draw.
"""

from __future__ import annotations

from contextlib import suppress

from zeus.device.machine.common import (
    EmptyMachine,
    Machine,
    MachineMeasurement,
    ZeusMachineInitError,
)
from zeus.device.machine.rpi import (
    RaspberryPi,
    RPIMeasurement,
    ZeusRaspberryPiInitError,
    rpi_is_available,
)

__all__ = [
    "EmptyMachine",
    "Machine",
    "MachineMeasurement",
    "RPIMeasurement",
    "RaspberryPi",
    "ZeusMachineInitError",
    "ZeusRaspberryPiInitError",
    "get_machine",
    "rpi_is_available",
]

_machine: Machine | None = None


def get_machine() -> Machine:
    """Initialize and return a singleton machine power meter.

    Currently supported:
        - Raspberry Pi 5

    Raises:
        ZeusMachineInitError: No machine-level meter could be initialized.
    """
    global _machine
    if _machine is not None:
        return _machine

    if rpi_is_available():
        with suppress(ZeusRaspberryPiInitError):
            _machine = RaspberryPi()

    # For additional machines, add more initialization attempts.
    if _machine is None:
        raise ZeusMachineInitError("No machine-level power meter is available on this host.")
    return _machine
