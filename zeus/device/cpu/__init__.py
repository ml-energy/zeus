"""Abstraction layer for CPU devices.

The main function of this module is [`get_cpus`][zeus.device.cpu.get_cpus],
which returns a CPU Manager object specific to the platform.
"""

from __future__ import annotations

import glob
import os
from typing import Literal

from zeus.device.cpu.common import CPUs, ZeusCPUInitError
from zeus.device.cpu.rapl import rapl_is_available, RAPLCPUs

_cpus: CPUs | None = None

# Sorted list of unique physical package (socket) IDs on the system, populated
# lazily on the first call to `get_current_cpu_index` and cached thereafter.
_package_ids: list[int] | None = None


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
        package_id = int(phys_package_file.read().strip())

    # Some platforms (e.g., ARM) use arbitrary identifiers rather than 0-based
    # socket indices for physical package IDs, so the socket index is the rank
    # of the package ID among all unique package IDs on the system. The set of
    # package IDs is scanned once and cached to avoid re-reading sysfs on every
    # call, which matters on machines with many cores.
    global _package_ids
    if _package_ids is None:
        package_ids = set()
        for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/physical_package_id"):
            try:
                with open(path) as f:
                    package_ids.add(int(f.read().strip()))
            except (OSError, ValueError):
                continue
        _package_ids = sorted(package_ids)

    # If the package ID was somehow missed by the scan, rank it against a local
    # copy without mutating the cache (which would pollute it with a queried ID).
    known_ids = _package_ids if package_id in _package_ids else sorted(set(_package_ids) | {package_id})

    return known_ids.index(package_id)


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
        raise ZeusCPUInitError("RAPL unvailable Failed to initialize CPU management library.")
