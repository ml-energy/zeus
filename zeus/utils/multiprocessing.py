"""Tools related to multiprocessing."""

from __future__ import annotations

import inspect
import logging
import multiprocessing
import warnings

logger = logging.getLogger(__name__)


def _is_being_called_at_module_level() -> bool:
    """Check if the current call is happening at module level (during import).

    This inspects the call stack to determine if the class constructor was
    called from module-level code (i.e., not inside any function or method).
    This is useful for detecting when objects are instantiated as global variables.

    The expected stack layout when called from warn_if_global_in_subprocess:
        0: _is_being_called_at_module_level
        1: warn_if_global_in_subprocess
        2: __init__ of the class (e.g., ZeusMonitor.__init__)
        3: The frame that called the constructor
           - If instantiated at module level: this is a <module> frame
           - If instantiated inside a function: this is the function's frame

    Returns:
        True if the call appears to be at module level, False otherwise.
    """
    stack = inspect.stack()

    # We need at least 4 frames to determine the caller of __init__
    if len(stack) < 4:
        return False

    # Check if the frame that called the constructor is at module level
    caller_of_init = stack[3]
    return caller_of_init.function == "<module>"


def warn_if_global_in_subprocess(class_name: str) -> None:
    """Warn if an object is being instantiated as a global variable in a subprocess.

    This detects a common pitfall where ZeusMonitor (or related classes) is
    instantiated as a global variable in a script. When the script spawns
    subprocesses using the "spawn" method (default on Windows and macOS),
    the subprocess re-imports the main module, causing:

    1. The global ZeusMonitor to be instantiated again in the subprocess
    2. Any other global initialization (e.g., loading DNN models) to run again
    3. Potential OOM errors, recursive process spawning, or other issues

    The solution is to guard instantiation with `if __name__ == "__main__":`
    or move it inside a function.

    Args:
        class_name: Name of the class being instantiated (for the warning message).
    """
    # Check if we're in a subprocess (not the main process)
    parent = multiprocessing.parent_process()
    if parent is None:
        return  # We're in the main process, no issue

    # Check if we're being instantiated at module level
    if not _is_being_called_at_module_level():
        return  # We're inside a function, which is fine

    # We're in a subprocess AND being instantiated at module level
    # This is the problematic pattern
    warnings.warn(
        f"{class_name} is being instantiated at module level (as a global variable) "
        f"in a child process spawned with the 'spawn' start method. This usually means "
        f"your main script has `{class_name.lower()} = {class_name}(...)` outside of any "
        f"function or `if __name__ == '__main__':` guard. When Zeus spawns subprocesses, "
        f"they re-import your main module, causing global initialization code to run again. "
        f"This can lead to OOM errors (e.g., if DNN models are loaded as globals), "
        f"recursive process spawning, or other unexpected behavior.\n\n"
        f"To fix this, either:\n"
        f"1. Move the {class_name} instantiation inside `if __name__ == '__main__':`\n"
        f"2. Move it inside a function that's called from the main block\n\n"
        f"See https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods",
        stacklevel=4,  # Point to the actual instantiation line
    )
