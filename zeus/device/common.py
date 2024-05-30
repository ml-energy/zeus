"""Common utilities for device management."""

from __future__ import annotations

import os
import ctypes
from functools import lru_cache

from zeus.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def has_sys_admin() -> bool:
    """Check if the current process has `SYS_ADMIN` capabilities."""
    # First try to read procfs.
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("CapEff"):
                    bitmask = int(line.strip().split()[1], 16)
                    has = bool(bitmask & (1 << 21))
                    logger.info(
                        "Read security capabilities from /proc/self/status -- SYS_ADMIN: %s",
                        has,
                    )
                    return has
    except Exception:
        logger.info("Failed to read capabilities from /proc/self/status", exc_info=True)

    # If that fails, try to use the capget syscall.
    class CapHeader(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class CapData(ctypes.Structure):
        _fields_ = [
            ("effective", ctypes.c_uint32),
            ("permitted", ctypes.c_uint32),
            ("inheritable", ctypes.c_uint32),
        ]

    # Attempt to load libc and set up capget
    try:
        libc = ctypes.CDLL("libc.so.6")
        capget = libc.capget
        capget.argtypes = [ctypes.POINTER(CapHeader), ctypes.POINTER(CapData)]
        capget.restype = ctypes.c_int
    except Exception:
        logger.info("Failed to load libc.so.6", exc_info=True)
        return False

    # Initialize the header and data structures
    header = CapHeader(version=0x20080522, pid=0)  # Use the current process
    data = CapData()

    # Call capget and check for errors
    if capget(ctypes.byref(header), ctypes.byref(data)) != 0:
        errno = ctypes.get_errno()
        logger.info(
            "capget failed with error: %s (errno %s)", os.strerror(errno), errno
        )
        return False

    bitmask = data.effective
    has = bool(bitmask & (1 << 21))
    logger.info("Read security capabilities from capget -- SYS_ADMIN: %s", has)
    return has
