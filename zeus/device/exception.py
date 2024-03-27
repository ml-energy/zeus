"""Base Zeus GPU Exception Class."""
from zeus.exception import ZeusBaseError


class ZeusBaseGPUError(ZeusBaseError):
    """Zeus base GPU exception class."""

    def __init__(self, message: str) -> None:
        """Initialize Base Zeus Exception."""
        super().__init__(message)
