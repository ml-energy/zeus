from __future__ import annotations

import builtins
import os
import pytest
from typing import Generator, TYPE_CHECKING, Sequence
from unittest.mock import patch, mock_open, create_autospec, MagicMock
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
    rapl_values = [
        "100000",
        "90000",
        "80000",
        "70000",
        "60000",
        "50000",
        "40000",
        "50000",
        "20000",
        "10000",
    ]
    mocked_rapl_file = MockRaplFile(RAPL_DIR + "/intel-rapl:0/energy_uj", rapl_values)
    mocked_rapl_file_name = mock_open()
    mocked_rapl_file_name.return_value.read.return_value = "package"
    mocked_rapl_file_max = mock_open()
    mocked_rapl_file_max.return_value.read.return_value = "100000"

    real_open = builtins.open

    def mock_file_open(filepath, *args, **kwargs):
        if filepath == (RAPL_DIR + "/intel-rapl:0/energy_uj"):
            return mocked_rapl_file
        if filepath == (RAPL_DIR + "/intel-rapl:0/name"):
            return mocked_rapl_file_name()
        if filepath == (RAPL_DIR + "/intel-rapl:0/max_energy_range_uj"):
            return mocked_rapl_file_max()
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


@pytest.fixture()
def mock_rapl_wraparound_tracker():
    patch_tracker = patch("zeus.device.cpu.rapl.RaplWraparoundTracker")
    MockRaplWraparoundTracker = patch_tracker.start()

    mock_tracker = MockRaplWraparoundTracker.return_value
    mock_tracker.get_num_wraparounds.side_effect = [0, 5]

    yield mock_tracker

    patch_tracker.stop()


def test_rapl_polling_process(mock_rapl_values):
    wraparound_counter = mp.Value("i", 0)
    with pytest.raises(MockRaplFileOutOfValues) as exception:
        _polling_process(RAPL_DIR + "/intel-rapl:0/energy_uj", 1000, wraparound_counter)
    assert wraparound_counter.value == 8


# RAPLFile tests
@pytest.fixture
@patch("os.path.exists", return_value=False)
def test_rapl_available(mock_exists):
    assert rapl_is_available() == False


def test_rapl_file_class(mock_rapl_values, mock_rapl_wraparound_tracker):
    """Test the `RAPLFile` class."""
    # Test initialization
    raplFile = RAPLFile("/sys/class/powercap/intel-rapl/intel-rapl:0")
    assert raplFile.name == "package"
    assert raplFile.last_energy == 100000.0
    assert raplFile.max_energy_range_uj == 100000.0

    # Test read method where get_num_wraparounds is 0
    assert raplFile.read() == 90.0

    # Test read method where get_num_wraparounds is 5
    assert raplFile.read() == 580.0  # (80000+5*100000)/1000


def test_rapl_file_class_exceptions():
    """Test `RAPLFile` Init errors"""
    with patch("builtins.open", mock_open()) as mock_file:
        # Fails to open name file
        mock_file.side_effect = FileNotFoundError
        with pytest.raises(ZeusRAPLFileInitError):
            RAPLFile("/sys/class/powercap/intel-rapl/intel-rapl:0")

        # Fails to read energy_uj file
        mock_file.side_effect = [
            mock_open(read_data="package").return_value,
            FileNotFoundError,
        ]
        with pytest.raises(ZeusRAPLFileInitError):
            RAPLFile("/sys/class/powercap/intel-rapl/intel-rapl:0")

        # Fails to read max_energy_uj file
        mock_file.side_effect = [
            mock_open(read_data="package").return_value,
            mock_open(read_data="1000000").return_value,
            FileNotFoundError,
        ]
        with pytest.raises(ZeusRAPLFileInitError):
            RAPLFile("/sys/class/powercap/intel-rapl/intel-rapl:0")


# RAPLCPU tests
@pytest.fixture()
def mock_os_listdir_cpu(mocker):
    return mocker.patch("os.listdir", return_value=["intel-rapl:0", "intel-rapl:0:0"])


def create_rapl_file_mock(name="package", read_value=1000.0):
    """Create a mock `RAPLFile` class"""
    mock_rapl_file = create_autospec(RAPLFile, instance=True)
    mock_rapl_file.name = name
    mock_rapl_file.read.return_value = read_value
    return mock_rapl_file


def test_rapl_cpu_class(mocker, mock_os_listdir_cpu):
    """Test `RAPLCPU` with `DRAM`"""
    mock_rapl_file_package = create_rapl_file_mock()
    mock_rapl_file_dram = create_rapl_file_mock(name="dram", read_value=500.0)

    def rapl_file_side_effect(path):
        if "0:0" in path:
            return mock_rapl_file_dram
        return mock_rapl_file_package

    mocker.patch("zeus.device.cpu.rapl.RAPLFile", side_effect=rapl_file_side_effect)
    cpu = RAPLCPU(cpu_index=0, rapl_dir=RAPL_DIR)
    measurement = cpu.getTotalEnergyConsumption()

    assert cpu.path == os.path.join(RAPL_DIR, "intel-rapl:0")
    assert cpu.rapl_file == mock_rapl_file_package
    assert cpu.dram == mock_rapl_file_dram
    assert measurement.cpu_mj == mock_rapl_file_package.read.return_value
    assert measurement.dram_mj == mock_rapl_file_dram.read.return_value


def test_rapl_cpu_class_exceptions(mocker, mock_os_listdir_cpu):
    """Test `RAPLCPU` subpackage init error"""
    mock_rapl_file_package = create_rapl_file_mock()
    mock_rapl_file_dram = create_rapl_file_mock(name="dram", read_value=500.0)

    def rapl_file_side_effect(path):
        if "0:0" in path:
            raise ZeusRAPLFileInitError("Initilization Error")
        return mock_rapl_file_package

    mocker.patch("zeus.device.cpu.rapl.RAPLFile", side_effect=rapl_file_side_effect)
    with warnings.catch_warnings(record=True) as w:
        cpu = RAPLCPU(cpu_index=0, rapl_dir=RAPL_DIR)
        assert "Failed to initialize subpackage" in str(w[-1].message)

    assert cpu.path == os.path.join(RAPL_DIR, "intel-rapl:0")
    assert cpu.rapl_file == mock_rapl_file_package
    assert cpu.dram is None


# RAPLCPUs tests
def test_rapl_cpus_class(mocker):
    """Test initialization when RAPL is available."""
    mocker.patch("zeus.device.cpu.rapl.rapl_is_available", return_value=True)
    mocker.patch(
        "zeus.device.cpu.rapl.glob",
        return_value=[f"{RAPL_DIR}/intel-rapl:0", f"{RAPL_DIR}/intel-rapl:1"],
    )
    mock_rapl_cpu_constructor = mocker.patch("zeus.device.cpu.rapl.RAPLCPU")
    mock_rapl_cpu_instance = MagicMock(spec=RAPLCPU)
    mock_rapl_cpu_constructor.side_effect = [
        mock_rapl_cpu_instance,
        mock_rapl_cpu_instance,
    ]
    rapl_cpus = RAPLCPUs()

    assert len(rapl_cpus.cpus) == 2
    assert all(isinstance(cpu, MagicMock) for cpu in rapl_cpus.cpus)
    assert mock_rapl_cpu_constructor.call_count == 2


def test_rapl_cpus_class_init_error(mocker):
    """Test initialization when RAPL is not available."""
    mocker.patch("zeus.device.cpu.rapl.rapl_is_available", return_value=False)

    with pytest.raises(
        ZeusRAPLNotSupportedError, match="RAPL is not supported on this CPU."
    ):
        RAPLCPUs()
