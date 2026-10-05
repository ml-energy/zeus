from __future__ import annotations

import builtins
import errno
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
    _read_zone_power_limits,
)
from zeus.device.cpu.common import CpuDramMeasurement
from zeus.utils.zeusd import CpuDramPowerLimits, PowerLimitConstraint, ZonePowerLimits


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


def test_rapl_file_lazily_initializes_wraparound_tracker():
    """Test that RAPLFile defers RaplWraparoundTracker creation until the first read()."""
    path = "/sys/class/powercap/intel-rapl/intel-rapl:0"

    mock_tracker_instance = MagicMock()
    mock_tracker_instance.get_num_wraparounds.return_value = 0

    mock_name_open = mock_open(read_data="package")
    mock_energy_open = mock_open(read_data="100000")
    mock_max_energy_open = mock_open(read_data="200000")

    def mock_file_open(filepath, *args, **kwargs):
        basename = os.path.basename(filepath)
        if basename == "name":
            return mock_name_open()
        if basename == "energy_uj":
            return mock_energy_open()
        if basename == "max_energy_range_uj":
            return mock_max_energy_open()
        raise FileNotFoundError(f"Unexpected file: {filepath}")

    with (
        patch("zeus.device.cpu.rapl.RaplWraparoundTracker") as mock_tracker_class,
        patch("builtins.open", side_effect=mock_file_open),
    ):
        mock_tracker_class.return_value = mock_tracker_instance

        rapl_file = RAPLFile(path)

        # Tracker must not be created during __init__
        mock_tracker_class.assert_not_called()
        assert rapl_file._wraparound_tracker is None

        # First read() must create the tracker exactly once
        rapl_file.read()
        mock_tracker_class.assert_called_once()
        assert rapl_file._wraparound_tracker is mock_tracker_instance

        # Subsequent read() calls must reuse the same tracker instance
        rapl_file.read()
        mock_tracker_class.assert_called_once()
        assert rapl_file._wraparound_tracker is mock_tracker_instance


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
    measurement = cpu.get_total_energy_consumption()

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


# Power limit tests
def write_rapl_zone(
    zone_dir: Path,
    name: str,
    enabled: str,
    constraints: Sequence[tuple[str, int, int, int]],
) -> None:
    """Write the sysfs files of a RAPL zone.

    Each constraint is `(name, power_limit_uw, max_power_uw, time_window_us)`.
    """
    zone_dir.mkdir(parents=True)
    (zone_dir / "name").write_text(f"{name}\n")
    (zone_dir / "energy_uj").write_text("1000\n")
    (zone_dir / "max_energy_range_uj").write_text("262143328850\n")
    (zone_dir / "enabled").write_text(f"{enabled}\n")
    for index, (constraint_name, power_limit_uw, max_power_uw, time_window_us) in enumerate(constraints):
        (zone_dir / f"constraint_{index}_name").write_text(f"{constraint_name}\n")
        (zone_dir / f"constraint_{index}_power_limit_uw").write_text(f"{power_limit_uw}\n")
        (zone_dir / f"constraint_{index}_max_power_uw").write_text(f"{max_power_uw}\n")
        (zone_dir / f"constraint_{index}_time_window_us").write_text(f"{time_window_us}\n")


PACKAGE_CONSTRAINTS = [
    ("long_term", 205_000_000, 205_000_000, 999_424),
    ("short_term", 246_000_000, 780_000_000, 999_424),
]
PACKAGE_LIMITS = ZonePowerLimits(
    enabled=True,
    constraints=[
        PowerLimitConstraint(name="long_term", power_limit_mw=205_000, max_power_mw=205_000, time_window_us=999_424),
        PowerLimitConstraint(name="short_term", power_limit_mw=246_000, max_power_mw=780_000, time_window_us=999_424),
    ],
)


def test_rapl_cpu_get_power_limits(tmp_path):
    """Test `RAPLCPU.get_power_limits` with package and DRAM zones."""
    package_dir = tmp_path / "intel-rapl:0"
    write_rapl_zone(package_dir, "package-0", "1", PACKAGE_CONSTRAINTS)
    write_rapl_zone(package_dir / "intel-rapl:0:0", "dram", "0", [("long_term", 0, 121_000_000, 976)])

    cpu = RAPLCPU(cpu_index=0, rapl_dir=str(tmp_path))

    assert cpu.get_power_limits() == CpuDramPowerLimits(
        cpu=PACKAGE_LIMITS,
        dram=ZonePowerLimits(
            enabled=False,
            constraints=[
                PowerLimitConstraint(name="long_term", power_limit_mw=0, max_power_mw=121_000, time_window_us=976),
            ],
        ),
    )


def test_rapl_cpu_get_power_limits_without_dram(tmp_path):
    """Test `RAPLCPU.get_power_limits` without a DRAM zone."""
    write_rapl_zone(tmp_path / "intel-rapl:0", "package-0", "1", PACKAGE_CONSTRAINTS)

    cpu = RAPLCPU(cpu_index=0, rapl_dir=str(tmp_path))

    assert cpu.get_power_limits() == CpuDramPowerLimits(cpu=PACKAGE_LIMITS, dram=None)


def test_read_zone_power_limits_without_constraints(tmp_path):
    """A zone without constraint files has an empty constraint list."""
    write_rapl_zone(tmp_path / "zone", "package-0", "0", [])

    assert _read_zone_power_limits(str(tmp_path / "zone")) == ZonePowerLimits(enabled=False, constraints=[])


def test_read_zone_power_limits_missing_file(tmp_path):
    """A constraint with a missing file raises an error."""
    zone_dir = tmp_path / "zone"
    write_rapl_zone(zone_dir, "package-0", "1", PACKAGE_CONSTRAINTS)
    (zone_dir / "constraint_1_max_power_uw").unlink()

    with pytest.raises(FileNotFoundError):
        _read_zone_power_limits(str(zone_dir))


def patch_open_to_fail(mocker, file: str, error: OSError) -> None:
    """Make `open` raise `error` for paths ending in `file`, and behave normally otherwise."""
    real_open = builtins.open

    def open_or_fail(path, *args, **kwargs):
        if str(path).endswith(file):
            raise error
        return real_open(path, *args, **kwargs)

    mocker.patch("builtins.open", side_effect=open_or_fail)


def test_read_zone_power_limits_enodata_attribute_is_none(tmp_path, mocker):
    """Kernels 6.5 and later answer `ENODATA` for the time window of `peak_power`."""
    zone_dir = tmp_path / "zone"
    write_rapl_zone(zone_dir, "package-0", "1", [*PACKAGE_CONSTRAINTS, ("peak_power", 300_000_000, 1_560_000_000, 0)])
    patch_open_to_fail(mocker, "constraint_2_time_window_us", OSError(errno.ENODATA, "No data available"))

    limits = _read_zone_power_limits(str(zone_dir))

    assert limits.constraints[:2] == PACKAGE_LIMITS.constraints
    assert limits.constraints[2] == PowerLimitConstraint(
        name="peak_power", power_limit_mw=300_000, max_power_mw=1_560_000, time_window_us=None
    )


@pytest.mark.parametrize(
    "file",
    [
        "enabled",
        "constraint_0_name",
        "constraint_0_power_limit_uw",
        "constraint_0_max_power_uw",
        "constraint_0_time_window_us",
    ],
)
def test_read_zone_power_limits_other_read_errors_propagate(tmp_path, mocker, file):
    """Only `ENODATA` is tolerated, and only for optional attributes."""
    zone_dir = tmp_path / "zone"
    write_rapl_zone(zone_dir, "package-0", "1", PACKAGE_CONSTRAINTS)
    patch_open_to_fail(mocker, file, PermissionError(errno.EACCES, "Permission denied"))

    with pytest.raises(PermissionError):
        _read_zone_power_limits(str(zone_dir))


def test_read_zone_power_limits_enodata_power_limit_is_error(tmp_path, mocker):
    """`power_limit_uw` is mandatory in powercap, so `ENODATA` on it is an error."""
    zone_dir = tmp_path / "zone"
    write_rapl_zone(zone_dir, "package-0", "1", PACKAGE_CONSTRAINTS)
    patch_open_to_fail(mocker, "constraint_0_power_limit_uw", OSError(errno.ENODATA, "No data available"))

    with pytest.raises(OSError):
        _read_zone_power_limits(str(zone_dir))


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

    with pytest.raises(ZeusRAPLNotSupportedError, match="RAPL is not supported on this CPU."):
        RAPLCPUs()
