"""Utilities for logging."""

import sys
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
