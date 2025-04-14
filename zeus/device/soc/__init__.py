"""Abstraction layer for SoC devices.

The main function of this module is [`get_soc`][zeus.device.soc.get_soc],
which returns a SoC Manager object specific to the platform.
"""

from __future__ import annotations

from zeus.device.soc.common import SoC, ZeusSoCInitError
from zeus.device.soc.jetson import Jetson

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
    
    try:
        # Instantiate the Jetson SoC if this is a Jetson device
        _soc = Jetson()
    except Exception as e:
        # If there's an error initializing the Jetson device, log or handle the error
        raise ZeusSoCInitError(f"Failed to initialize Jetson SoC: {str(e)}")

    # SoCs in the future can be incorporated via `elif` blocks.
    else:
        # Placeholder to avoid linting error (remove once _soc can be assigned a real value).
        raise ZeusSoCInitError("No observable SoC was found on the current machine.")
