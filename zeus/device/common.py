"""Common utilities for device management."""

from __future__ import annotations

import abc
import os
import ctypes
import functools
import logging
import warnings
from typing import Callable
from functools import lru_cache

logger = logging.getLogger(__name__)


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
        logger.info("capget failed with error: %s (errno %s)", os.strerror(errno), errno)
        return False

    bitmask = data.effective
    has = bool(bitmask & (1 << 21))
    logger.info("Read security capabilities from capget -- SYS_ADMIN: %s", has)
    return has


def deprecated_alias(old_name: str) -> Callable[[Callable], Callable]:
    """Decorator that marks a method to have a deprecated camelCase alias.

    Apply this decorator to the new snake_case method. When the old camelCase
    name is called, it will emit a deprecation warning once and then call the
    new snake_case method.

    Example:
        ```python
        @deprecated_alias("getName")
        def get_name(self):
            return "GPU Name"
        ```

    The class using this decorator should use `DeprecatedAliasABCMeta` as its metaclass.

    Args:
        old_name: The old camelCase method name to create as a deprecated alias.

    Returns:
        The decorated function with the `_deprecated_alias` attribute set.
    """

    def decorator(func):
        func._deprecated_alias = old_name
        return func

    return decorator


def _make_deprecated_method(new_method: Callable, old_name: str, new_name: str) -> Callable:
    """Create a deprecated method wrapper that warns once globally per method.

    Args:
        new_method: The new snake_case method to be called.
        old_name: The old camelCase method name (for the warning message).
        new_name: The new snake_case method name (for the warning message).

    Returns:
        A wrapper function that emits a deprecation warning and calls the new method.
    """
    warned = [False]

    @functools.wraps(new_method)
    def deprecated_method(self, *args, **kwargs):
        if not warned[0]:
            warnings.warn(
                f"'{old_name}' is deprecated and will be removed in a future version. Use '{new_name}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            warned[0] = True
        return getattr(self, new_name)(*args, **kwargs)

    # Remove the _deprecated_alias attribute to prevent recursion when the metaclass
    # processes this deprecated method during class creation
    if hasattr(deprecated_method, "_deprecated_alias"):
        delattr(deprecated_method, "_deprecated_alias")

    return deprecated_method


class DeprecatedAliasABCMeta(abc.ABCMeta):
    """Metaclass that combines ABC functionality with automatic deprecated alias creation.

    This metaclass looks for methods decorated with `@deprecated_alias` and automatically
    creates the old camelCase method names that emit deprecation warnings once and then
    call the new snake_case methods.

    Since this is frequently composed with `abc.ABCMeta`, this metaclass inherits from it
    to avoid metaclass conflicts.

    !!! Example
        ```python
        class MyClass(abc.ABC, metaclass=DeprecatedAliasABCMeta):
            @deprecated_alias("oldMethod")
            @abc.abstractmethod
            def new_method(self):
                pass

        class MyImplementation(MyClass):
            def new_method(self):
                return "implementation"

        obj = MyImplementation()
        obj.new_method()  # No warning
        obj.oldMethod()   # Emits deprecation warning, calls new_method
        ```
    """

    def __new__(mcs, name, bases, namespace):
        """Create the class and add deprecated alias methods."""
        cls = super().__new__(mcs, name, bases, namespace)

        # Create deprecated aliases for methods marked with @deprecated_alias
        for attr_name in dir(cls):
            if attr_name.startswith("_"):
                continue

            try:
                attr = getattr(cls, attr_name)
            except AttributeError:
                continue

            # Check if this method has the deprecated alias marker
            if hasattr(attr, "_deprecated_alias"):
                old_name = attr._deprecated_alias
                # Create and attach the deprecated wrapper method
                deprecated_method = _make_deprecated_method(attr, old_name, attr_name)
                setattr(cls, old_name, deprecated_method)

        return cls
