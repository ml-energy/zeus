"""Base Zeus Exception Class."""


class ZeusBaseError(Exception):
    """Zeus base exception class."""

    def __init__(self, message: str) -> None:
        """Initialize Base Zeus Exception."""
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        """Return message."""
        return self.message
