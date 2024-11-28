"""Tools related to environment variables."""

from __future__ import annotations

import os
from typing import Type, TypeVar, cast

T = TypeVar("T")


def get_env(name: str, valtype: Type[T], default: T | None = None) -> T:
    """Fetch an environment variable and cast it to the given type."""
    try:
        if valtype is bool:
            val = os.environ[name].lower()
            if val not in ["true", "false"]:
                raise ValueError(f"Strange boolean environment variable value '{val}'")
            return cast(T, val == "true")
        return valtype(os.environ[name])  # type: ignore
    except KeyError:
        if default is not None:
            return default
        raise ValueError(f"Missing environment variable '{name}'") from None
