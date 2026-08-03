"""Abstraction layer for CPU devices.

The main function of this module is [`get_cpus`][zeus.device.cpu.get_cpus],
which returns a CPU Manager object specific to the platform.
"""

from __future__ import annotations

from zeus.device.cpu.common import CPUs, ZeusCPUInitError
from zeus.device.cpu.emi import EMICPUs, emi_is_available
from zeus.device.cpu.rapl import rapl_is_available, RAPLCPUs

_cpus: CPUs | None = None


def get_cpus() -> CPUs:
    """Initialize and return a singleton CPU monitoring object for Intel CPUs.

    The function returns a CPU management object that abstracts the underlying
    CPU energy monitoring interface: EMI on Windows, RAPL on Linux.

    Raises:
        ZeusCPUInitError: If no supported CPU energy monitoring interface is available.
    """
    global _cpus
    if _cpus is not None:
        return _cpus
    if emi_is_available():
        _cpus = EMICPUs()
        return _cpus
    if rapl_is_available():
        _cpus = RAPLCPUs()
        return _cpus
    raise ZeusCPUInitError(
        "No supported CPU energy monitoring interface is available. "
        "EMI requires Windows 10+ with an Intel EMI-compatible driver. "
        "RAPL requires Linux with the intel-rapl kernel module."
    )
