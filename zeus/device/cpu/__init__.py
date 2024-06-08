"""Abstraction layer for CPU devices.

The main function of this module is [`get_cpus`][zeus.device.cpu.get_cpus],
which returns a CPU Manager object specific to the platform.
"""

from __future__ import annotations

from zeus.device.cpu.common import CPUs, ZeusCPUInitError
from zeus.device.cpu.rapl import rapl_is_available, RAPLCPUs

_cpus: CPUs | None = None


def get_cpus() -> CPUs:
    """Initialize and return a singleton CPU monitoring object for INTEL CPUs.

    The function returns a CPU management object that aims to abstract the underlying CPU monitoring libraries
    (RAPL for Intel CPUs).

    This function attempts to initialize CPU mointoring using RAPL. If this attempt fails, it raises
    a ZeusErrorInit exception.
    """
    global _cpus
    if _cpus is not None:
        return _cpus
    if rapl_is_available():
        _cpus = RAPLCPUs()
        return _cpus
    else:
        raise ZeusCPUInitError(
            "RAPL unvailable Failed to initialize CPU management library."
        )
