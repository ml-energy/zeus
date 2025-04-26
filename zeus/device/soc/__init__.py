"""Abstraction layer for SoC devices.

The main function of this module is [`get_soc`][zeus.device.soc.get_soc],
which returns a SoC Manager object specific to the platform.
"""

from __future__ import annotations

from contextlib import suppress

from zeus.device.soc.common import SoC, ZeusSoCInitError
from zeus.device.soc.jetson import Jetson, ZeusJetsonInitError, jetson_is_available
from zeus.device.soc.apple import (
    AppleSilicon,
    ZeusAppleInitError,
    apple_silicon_is_available,
)

_soc: SoC | None = None


def get_soc() -> SoC:
    """Initialize and return a singleton monolithic SoC monitoring object.

    The function returns a SoC management object that aims to abstract underlying SoC monitoring
    functionalities.

    Currently supported SoC devices:
        - Apple Silicon

    If no SoC monitor object can be initialized, a `ZeusSoCInitError` exception will be raised.
    """
    global _soc
    if _soc is not None:
        return _soc

    # --- Apple Silicon ---
    if apple_silicon_is_available():
        with suppress(ZeusAppleInitError):
            _soc = AppleSilicon()

    # --- Jetson Nano ---
    elif jetson_is_available():
        with suppress(ZeusJetsonInitError):
            _soc = Jetson()

    # For additional SoC's, add more initialization attempts.
    if _soc is None:
        raise ZeusSoCInitError("No observable SoC was found on the current machine.")
    return _soc
