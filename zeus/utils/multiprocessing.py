"""Multiprocessing-related utilities."""

from __future__ import annotations

import inspect
import multiprocessing as mp
import sys
import warnings
from typing import Any


def _is_spawned_child() -> bool:
    """Return True if running in a spawned child process.

    This checks both:
    1. Whether we have a parent process (we're a subprocess)
    2. Whether __mp_main__ is in sys.modules (indicates spawn method was used)
    """
    return mp.parent_process() is not None and "__mp_main__" in sys.modules


def _called_from_module_level() -> bool:
    """Return True if any caller frame is executing module-level code in a real file.

    This walks the call stack looking for any <module> frame from a real Python
    file (not multiprocessing infrastructure like <string> or <frozen ...>).
    Such a frame indicates that code is being executed at module level during import.
    """
    frame = inspect.currentframe()
    if frame is None:
        return False
    frame = frame.f_back
    while frame is not None:
        if frame.f_code.co_name == "<module>":
            filename = frame.f_code.co_filename
            # Skip multiprocessing infrastructure frames
            if not (filename == "<string>" or filename.startswith("<frozen")):
                return True
        frame = frame.f_back
    return False


def warn_if_global_in_subprocess(self: Any) -> None:
    """Warn when a monitor is created at import time in a spawned subprocess.

    This detects a common pitfall where ZeusMonitor (or related classes) is
    instantiated as a global variable or called from module-level code.
    When the script spawns subprocesses using the "spawn" method, the
    subprocess re-imports the main module, causing global initialization
    code to run again (e.g., loading DNN models), leading to OOM errors.

    Args:
        self: The instance being constructed (used to derive class name).
    """
    if not _is_spawned_child():
        return
    if not _called_from_module_level():
        return
    class_name = type(self).__name__
    warnings.warn(
        f"{class_name} was instantiated during module import in a spawned subprocess. "
        "This usually means the monitor was created as a global variable, so the child "
        "process re-imported your main module and executed global code again. Move monitor "
        'construction under `if __name__ == "__main__":` or inside a function to avoid '
        "repeated imports and memory issues.",
        stacklevel=4,
    )
