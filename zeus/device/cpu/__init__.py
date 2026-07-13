"""Abstraction layer for CPU devices.

The main function of this module is [`get_cpus`][zeus.device.cpu.get_cpus],
which returns a CPU Manager object specific to the platform.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Literal

from zeus.device.cpu.common import CPUs, ZeusCPUInitError
from zeus.device.cpu.emi import EMICPUs, emi_is_available
from zeus.device.cpu.rapl import RAPLCPUs, rapl_is_available

logger = logging.getLogger(__name__)

_cpus: CPUs | None = None


def get_current_cpu_index(pid: int | Literal["current"] = "current") -> int:
    """Retrieves the specific CPU index (socket) where the given PID is running.

    If no PID is given or pid is "current", the CPU index returned is of the CPU running the current process.

    On Linux, the index is read from ``/proc/{pid}/stat`` and the sysfs topology files.
    On Windows, the index is determined using the Windows logical processor information API.

    !!! Note
        Linux schedulers can preempt and reschedule processes to different CPUs. To prevent this from happening
        during monitoring, use `taskset` to pin processes to specific CPUs.
    """
    if sys.platform == "win32":
        return _get_current_cpu_index_windows()

    if pid == "current":
        pid = os.getpid()

    with open(f"/proc/{pid}/stat") as stat_file:
        cpu_core = int(stat_file.read().split()[38])

    with open(f"/sys/devices/system/cpu/cpu{cpu_core}/topology/physical_package_id") as phys_package_file:
        return int(phys_package_file.read().strip())


def _get_current_cpu_index_windows() -> int:
    """Return the CPU package index for the currently executing thread on Windows.

    Uses ``GetCurrentProcessorNumber`` to identify the logical processor and
    ``GetLogicalProcessorInformationEx`` with ``RelationProcessorPackage`` to
    map it to a physical package (socket) index.
    """
    import ctypes
    import ctypes.wintypes
    from ctypes import windll, wintypes

    kernel32 = windll.kernel32
    kernel32.GetCurrentProcessorNumber.restype = wintypes.DWORD
    kernel32.GetLogicalProcessorInformationEx.restype = wintypes.BOOL

    current_processor = kernel32.GetCurrentProcessorNumber()

    # Obtain the size of the buffer required for RelationProcessorPackage (= 3).
    relation_processor_package = 3
    buf_size = wintypes.DWORD(0)
    kernel32.GetLogicalProcessorInformationEx(relation_processor_package, None, ctypes.byref(buf_size))

    buf = ctypes.create_string_buffer(buf_size.value)
    if not kernel32.GetLogicalProcessorInformationEx(relation_processor_package, buf, ctypes.byref(buf_size)):
        logger.warning("GetLogicalProcessorInformationEx failed; defaulting to CPU package index 0.")
        return 0

    # Parse the variable-length SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX records.
    # Each record starts with:
    #   DWORD Relationship  (4 bytes)
    #   DWORD Size          (4 bytes)
    # followed by PROCESSOR_RELATIONSHIP:
    #   BYTE  Flags             (1)
    #   BYTE  EfficiencyClass   (1)
    #   BYTE  Reserved[20]     (20)
    #   WORD  GroupCount        (2)
    #   GROUP_AFFINITY GroupMask[GroupCount]
    # Each GROUP_AFFINITY:
    #   ULONG_PTR Mask  (8 bytes on 64-bit)
    #   WORD Group      (2)
    #   WORD Reserved[3](6)
    #   total = 16 bytes on 64-bit

    raw = bytes(buf)
    offset = 0
    package_index = 0
    ptr_size = ctypes.sizeof(ctypes.c_void_p)  # 8 on 64-bit, 4 on 32-bit
    group_affinity_size = ptr_size + 2 + 6  # Mask + Group + Reserved[3]

    while offset < buf_size.value:
        record_size = int.from_bytes(raw[offset + 4 : offset + 8], "little")

        # PROCESSOR_RELATIONSHIP starts at offset + 8.
        # GroupCount is at offset 22 within PROCESSOR_RELATIONSHIP (1+1+20 = 22).
        group_count = int.from_bytes(raw[offset + 8 + 22 : offset + 8 + 24], "little")

        # GROUP_AFFINITY array starts at offset 24 within PROCESSOR_RELATIONSHIP.
        for g in range(group_count):
            ga_offset = offset + 8 + 24 + g * group_affinity_size
            mask = int.from_bytes(raw[ga_offset : ga_offset + ptr_size], "little")
            if (mask >> current_processor) & 1:
                return package_index

        package_index += 1
        offset += record_size

    logger.warning(
        "Could not map logical processor %d to a package; defaulting to 0.",
        current_processor,
    )
    return 0


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
