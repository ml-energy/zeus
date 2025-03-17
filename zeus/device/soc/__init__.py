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

    Currently, no management object has been implemented for any SoC architecture, so calling this
    function will raise a `ZeusSoCInitError` error; implementations for SoC devices are expected
    to be added in the near future.
    """
    global _soc
    if _soc is not None:
        return _soc

    # SoCs in the future can be incorporated via `elif` blocks.
    else:
        # Placeholder to avoid linting error (remove once _soc can be assigned a real value).
        _soc = None

        raise ZeusSoCInitError("No observable SoC was found on the current machine.")
