"""Abstraction layer for SoC devices.

The main function of this module is [`get_soc`][zeus.device.soc.get_soc],
which returns a SoC Manager object specific to the platform.
"""

from __future__ import annotations

from zeus.device.soc.common import SoC, ZeusSoCInitError

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
    try:
        # The below import is done here instead of at top of file because the
        # Apple Silicon module contains optional dependencies that will cause
        # import failures if not installed on the host device.
        from zeus.device.soc.apple import AppleSilicon, ZeusAppleInitError

        _soc = AppleSilicon()
    except (ImportError, ZeusSoCInitError):
        pass

    # For additional SoC's, add more initialization attempts.

    if _soc is None:
        raise ZeusSoCInitError("No observable SoC was found on the current machine.")
    return _soc
