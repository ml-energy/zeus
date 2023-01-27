# Copyright (C) 2022 Jae-Won Chung <jwnchung@umich.edu>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for logging."""

import sys
import logging
from pathlib import Path


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
LOG_HANDLER = logging.StreamHandler()
LOG_FORMATTER = logging.Formatter("%(asctime)s %(message)s")
LOG_HANDLER.setFormatter(LOG_FORMATTER)
LOG.addHandler(LOG_HANDLER)


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
