"""Multiprocessing-related utilities."""

from __future__ import annotations

import inspect
import warnings
from typing import Any


def _is_global_in_spawned_child() -> bool:
    """Return True if called from module-level code in a spawned child's main script.

    In a spawned child process (using the "spawn" start method), the main script
    is re-imported with `__name__ = "__mp_main__"` instead of `"__main__"`. This
    function detects if we're currently executing module-level code (like global
    variable initialization) in such a script.

    This walks the call stack looking for any <module> frame from a real Python
    file (not multiprocessing infrastructure like <string> or <frozen ...>) where
    the module's `__name__` is `"__mp_main__"`.
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
                module_name = frame.f_globals.get("__name__", "")
                if module_name == "__mp_main__":
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
    if not _is_global_in_spawned_child():
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
