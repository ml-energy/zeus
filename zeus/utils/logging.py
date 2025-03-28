"""Utilities for logging."""

import os
import sys
import logging
from pathlib import Path


class FileAndConsole:
    """Like tee, but for Python prints."""

    def __init__(self, filepath: Path) -> None:
        """Initialize the object."""
        self.file = open(filepath, "w")
        self.stdout = sys.stdout

    def write(self, message):
        """Write message."""
        self.file.write(message)
        self.stdout.write(message)
        self.file.flush()
        self.stdout.flush()

    def flush(self):
        """Flush both log file and stdout."""
        self.file.flush()
        self.stdout.flush()


def get_logger(
    name: str,
    level: int = logging.INFO,
    propagate: bool = False,
) -> logging.Logger:
    """Get a logger with the given name with some formatting configs."""
    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.propagate = propagate
    logger.setLevel(os.environ.get("ZEUS_LOG_LEVEL", level))
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s](%(filename)s:%(lineno)d) %(message)s"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
