"""Abstraction layer for CPU devices.

The main function of this module is [`get_cpus`][zeus.device.cpu.get_cpus],
which returns a CPU Manager object specific to the platform.
"""

from __future__ import annotations

import os
from typing import Literal

from zeus.device.cpu.common import CPUs, ZeusCPUInitError
from zeus.device.cpu.emi import EMICPUs, emi_is_available
from zeus.device.cpu.rapl import rapl_is_available, RAPLCPUs

_cpus: CPUs | None = None


def get_current_cpu_index(pid: int | Literal["current"] = "current") -> int:
    """Retrieves the specific CPU index (socket) where the given PID is running.

    If no PID is given or pid is "current", the CPU index returned is of the CPU running the current process.

    !!! Note
        Linux schedulers can preempt and reschedule processes to different CPUs. To prevent this from happening
        during monitoring, use `taskset` to pin processes to specific CPUs.
    """
    if pid == "current":
        pid = os.getpid()

    with open(f"/proc/{pid}/stat") as stat_file:
        cpu_core = int(stat_file.read().split()[38])

    with open(f"/sys/devices/system/cpu/cpu{cpu_core}/topology/physical_package_id") as phys_package_file:
        return int(phys_package_file.read().strip())


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
