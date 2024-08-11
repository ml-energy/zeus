from __future__ import annotations

import builtins
import os
import pytest
from typing import Generator, TYPE_CHECKING, Sequence
from time import sleep
from unittest.mock import patch, mock_open, create_autospec, MagicMock
import unittest.mock as mock
from sys import stdout, stderr
import warnings

import multiprocessing as mp

if TYPE_CHECKING:
    from pathlib import Path

from zeus.device.cpu.rapl import (
    RAPLFile,
    RAPLCPU,
    RAPLCPUs,
    ZeusRAPLFileInitError,
    ZeusRAPLNotSupportedError,
    rapl_is_available,
    RAPL_DIR,
    RaplWraparoundTracker,
    _polling_process,
)
from zeus.device.cpu.common import CpuDramMeasurement

class MockRaplFileOutOfValues(Exception):
    """Exception raised when MockRaplFile runs out of values."""

    def __init__(self, message="Out of values"):
        self.message = message

class MockRaplFile:
    def __init__(self, file_path, values):
        self.file_path = file_path
        self.values = iter(values)
    
    def read(self, *args, **kwargs):
        if (value := next(self.values, None)) is not None:
            return value
        raise MockRaplFileOutOfValues()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

@pytest.fixture
def mock_rapl_values():
    rapl_values = ["1000", "900", "800", "700", "600", "500", "400", "500", "200", "100"]
    mocked_rapl_file = MockRaplFile(RAPL_DIR+"/intel-rapl:0/energy_uj", rapl_values)

    real_open = builtins.open

    def mock_file_open(filepath, *args, **kwargs):
        if filepath == (RAPL_DIR+"/intel-rapl:0/energy_uj"):
            return mocked_rapl_file
        else:
            return real_open(filepath, *args, **kwargs)

    patch_exists = patch("os.path.exists", return_value=True)
    patch_open = patch("builtins.open", side_effect=mock_file_open)
    patch_sleep = patch("time.sleep", return_value=None)

    patch_exists.start()
    patch_open.start()
    patch_sleep.start()

    yield

    patch_exists.stop()
    patch_open.stop()
    patch_sleep.stop()

def test_rapl_polling_process(mock_rapl_values):
    wraparound_counter = mp.Value('i', 0)
    with pytest.raises(MockRaplFileOutOfValues) as exception:
        _polling_process(RAPL_DIR+"/intel-rapl:0/energy_uj", 1000, wraparound_counter)
    assert(wraparound_counter.value == 8)

