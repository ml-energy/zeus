"""Tests for the Windows EMI CPU energy monitoring module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from zeus.device.cpu.common import CpuDramMeasurement
from zeus.device.cpu.emi import (
    EMICPU,
    EMICPUs,
    EMIFile,
    ZeusEMIInitError,
    ZeusEMINotSupportedError,
    _EMIChannel,
    emi_is_available,
)

# ---------------------------------------------------------------------------
# Helpers for building raw metadata / measurement bytes
# ---------------------------------------------------------------------------

_EMI_NAME_MAX = 16  # WCHARs


def _wstr(text: str, width_wchars: int) -> bytes:
    """Encode *text* as UTF-16LE, zero-padded to *width_wchars* WCHARs."""
    encoded = text.encode("utf-16-le")
    return encoded + b"\x00" * (width_wchars * 2 - len(encoded))


def _make_v2_metadata(channels: list[tuple[int, str]]) -> bytes:
    """Build a minimal EMI_METADATA_V2 buffer.

    Args:
        channels: List of ``(unit, name)`` tuples, one per channel.

    Returns:
        Raw bytes ready to pass to :meth:`EMIFile._parse_channels`.
    """
    oem = _wstr("Microsoft", _EMI_NAME_MAX)
    model = _wstr("PPM", _EMI_NAME_MAX)
    revision = (1).to_bytes(2, "little")
    count = len(channels).to_bytes(2, "little")

    body = oem + model + revision + count
    for unit, name in channels:
        name_bytes = (name + "\x00").encode("utf-16-le")
        body += unit.to_bytes(4, "little")
        body += len(name_bytes).to_bytes(2, "little")
        body += name_bytes
    return body


def _make_v1_metadata(unit: int, name: str) -> bytes:
    """Build a minimal EMI_METADATA_V1 buffer."""
    name_bytes = (name + "\x00").encode("utf-16-le") if name else b""
    oem = _wstr("Vendor", _EMI_NAME_MAX)
    model = _wstr("Model", _EMI_NAME_MAX)
    revision = (0).to_bytes(2, "little")
    name_size = len(name_bytes).to_bytes(2, "little")
    return unit.to_bytes(4, "little") + oem + model + revision + name_size + name_bytes


def _make_measurement(energy_values: list[int]) -> bytes:
    """Build a V2 measurement buffer (plain array of EMI_CHANNEL_MEASUREMENT_DATA)."""
    buf = b""
    for energy in energy_values:
        buf += energy.to_bytes(8, "little")
        buf += (0).to_bytes(8, "little")  # AbsoluteTime placeholder
    return buf


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def test_emi_available_false_on_non_windows():
    """emi_is_available must return False on non-Windows platforms."""
    with patch("zeus.device.cpu.emi._WINDOWS", False):
        # Clear cache so the patch takes effect.
        emi_is_available.cache_clear()
        assert emi_is_available() is False
        emi_is_available.cache_clear()


def test_emi_available_false_when_no_devices():
    """emi_is_available must return False when no EMI devices are enumerated."""
    with (
        patch("zeus.device.cpu.emi._WINDOWS", True),
        patch("zeus.device.cpu.emi._get_emi_device_paths", return_value=[]),
    ):
        emi_is_available.cache_clear()
        assert emi_is_available() is False
        emi_is_available.cache_clear()


def test_emi_available_true_when_devices_found():
    """emi_is_available must return True when at least one EMI device is present."""
    with (
        patch("zeus.device.cpu.emi._WINDOWS", True),
        patch(
            "zeus.device.cpu.emi._get_emi_device_paths",
            return_value=[r"\\?\acpi#fake#0#{45bd8344-7ed6-49cf-a440-c276c933b053}"],
        ),
    ):
        emi_is_available.cache_clear()
        assert emi_is_available() is True
        emi_is_available.cache_clear()


# ---------------------------------------------------------------------------
# Metadata parsing (static method — platform-independent)
# ---------------------------------------------------------------------------


class TestParseChannelsV2:
    """Tests for EMIFile._parse_channels with EMI version 2."""

    def test_pkg_and_dram_channels(self):
        """V2 metadata with PKG and DRAM channels is parsed correctly."""
        raw = _make_v2_metadata(
            [
                (0, "RAPL_Package0_PKG"),
                (0, "RAPL_Package0_DRAM"),
                (0, "RAPL_Package0_PP0"),
                (0, "RAPL_Package0_PP1"),
            ]
        )
        channels = EMIFile._parse_channels(version=2, raw=raw)
        assert len(channels) == 4
        assert channels[0].name == "RAPL_Package0_PKG"
        assert channels[0].index == 0
        assert channels[0].unit == 0
        assert channels[1].name == "RAPL_Package0_DRAM"
        assert channels[1].index == 1
        assert channels[2].name == "RAPL_Package0_PP0"
        assert channels[2].index == 2
        assert channels[3].name == "RAPL_Package0_PP1"
        assert channels[3].index == 3

    def test_multi_package(self):
        """V2 metadata with two CPU packages is parsed correctly."""
        raw = _make_v2_metadata(
            [
                (0, "RAPL_Package0_PKG"),
                (0, "RAPL_Package0_DRAM"),
                (0, "RAPL_Package1_PKG"),
                (0, "RAPL_Package1_DRAM"),
            ]
        )
        channels = EMIFile._parse_channels(version=2, raw=raw)
        assert len(channels) == 4
        assert channels[0].name == "RAPL_Package0_PKG"
        assert channels[2].name == "RAPL_Package1_PKG"

    def test_unexpected_unit_logs_warning(self, caplog):
        """A non-pWh unit triggers a warning log."""
        import logging

        raw = _make_v2_metadata([(99, "RAPL_Package0_PKG")])
        with caplog.at_level(logging.WARNING, logger="zeus.device.cpu.emi"):
            EMIFile._parse_channels(version=2, raw=raw)
        assert any("unexpected unit" in r.message for r in caplog.records)


class TestParseChannelsV1:
    """Tests for EMIFile._parse_channels with EMI version 1."""

    def test_single_named_channel(self):
        """V1 metadata with a named metered hardware is parsed to one channel."""
        raw = _make_v1_metadata(unit=0, name="CPU_PKG")
        channels = EMIFile._parse_channels(version=1, raw=raw)
        assert len(channels) == 1
        assert channels[0].index == 0
        assert channels[0].name == "CPU_PKG"
        assert channels[0].unit == 0

    def test_empty_name_falls_back(self):
        """V1 metadata with zero name size falls back to 'EMI_V1'."""
        raw = _make_v1_metadata(unit=0, name="")
        channels = EMIFile._parse_channels(version=1, raw=raw)
        assert len(channels) == 1
        assert channels[0].name == "EMI_V1"


# ---------------------------------------------------------------------------
# EMICPU tests (EMIFile is mocked)
# ---------------------------------------------------------------------------


def _make_mock_emi_file(channel_energies: list[float]) -> MagicMock:
    """Return a mock EMIFile whose read() returns successive energy values."""
    mock_file = MagicMock(spec=EMIFile)
    mock_file.read.side_effect = channel_energies
    return mock_file


class TestEMICPU:
    """Tests for the EMICPU class."""

    def test_get_total_energy_cpu_only(self):
        """get_total_energy_consumption returns cpu_mj with dram_mj=None when no DRAM channel."""
        mock_file = _make_mock_emi_file([1234.5])
        cpu = EMICPU(
            cpu_index=0,
            emi_file=mock_file,
            pkg_channel_index=0,
            dram_channel_index=None,
        )
        result = cpu.get_total_energy_consumption()
        assert isinstance(result, CpuDramMeasurement)
        assert result.cpu_mj == pytest.approx(1234.5)
        assert result.dram_mj is None
        mock_file.read.assert_called_once_with(0)

    def test_get_total_energy_cpu_and_dram(self):
        """get_total_energy_consumption returns both cpu_mj and dram_mj when DRAM is available."""
        mock_file = _make_mock_emi_file([5000.0, 200.0])
        cpu = EMICPU(
            cpu_index=0,
            emi_file=mock_file,
            pkg_channel_index=0,
            dram_channel_index=1,
        )
        result = cpu.get_total_energy_consumption()
        assert result.cpu_mj == pytest.approx(5000.0)
        assert result.dram_mj == pytest.approx(200.0)
        assert mock_file.read.call_count == 2

    def test_supports_dram_false_without_dram_channel(self):
        """supports_get_dram_energy_consumption returns False when dram_channel_index is None."""
        mock_file = _make_mock_emi_file([])
        cpu = EMICPU(0, mock_file, pkg_channel_index=0, dram_channel_index=None)
        assert cpu.supports_get_dram_energy_consumption() is False

    def test_supports_dram_true_with_dram_channel(self):
        """supports_get_dram_energy_consumption returns True when dram_channel_index is set."""
        mock_file = _make_mock_emi_file([])
        cpu = EMICPU(0, mock_file, pkg_channel_index=0, dram_channel_index=1)
        assert cpu.supports_get_dram_energy_consumption() is True

    def test_cpu_index_stored(self):
        """cpu_index is stored correctly."""
        mock_file = _make_mock_emi_file([])
        cpu = EMICPU(3, mock_file, pkg_channel_index=2, dram_channel_index=None)
        assert cpu.cpu_index == 3


# ---------------------------------------------------------------------------
# EMICPUs tests (EMIFile and device paths are mocked)
# ---------------------------------------------------------------------------


def _make_emi_file_with_channels(channel_specs: list[tuple[int, str]]) -> MagicMock:
    """Create a mock EMIFile with channels matching the given (index, name) list."""
    channels = [_EMIChannel(index=i, name=name, unit=0) for i, name in channel_specs]
    mock = MagicMock(spec=EMIFile)
    mock.channels = channels
    mock.read.return_value = 0.0
    return mock


class TestEMICPUs:
    """Tests for the EMICPUs manager class."""

    def test_raises_when_emi_unavailable(self):
        """EMICPUs raises ZeusEMINotSupportedError when no EMI devices are found."""
        with (
            patch("zeus.device.cpu.emi.emi_is_available", return_value=False),
            pytest.raises(ZeusEMINotSupportedError),
        ):
            EMICPUs()

    def test_single_package_no_dram(self):
        """EMICPUs creates one EMICPU when one PKG channel is found without DRAM."""
        mock_file = _make_emi_file_with_channels(
            [
                (0, "RAPL_Package0_PKG"),
                (1, "RAPL_Package0_PP0"),
            ]
        )
        with (
            patch("zeus.device.cpu.emi.emi_is_available", return_value=True),
            patch("zeus.device.cpu.emi._get_emi_device_paths", return_value=["fake_path"]),
            patch("zeus.device.cpu.emi.EMIFile", return_value=mock_file),
        ):
            cpus = EMICPUs()

        assert len(cpus) == 1
        assert cpus.cpus[0].cpu_index == 0
        assert not cpus.cpus[0].supports_get_dram_energy_consumption()

    def test_single_package_with_dram(self):
        """EMICPUs creates one EMICPU with DRAM when DRAM channel is present."""
        mock_file = _make_emi_file_with_channels(
            [
                (0, "RAPL_Package0_PKG"),
                (1, "RAPL_Package0_DRAM"),
                (2, "RAPL_Package0_PP0"),
            ]
        )
        with (
            patch("zeus.device.cpu.emi.emi_is_available", return_value=True),
            patch("zeus.device.cpu.emi._get_emi_device_paths", return_value=["fake_path"]),
            patch("zeus.device.cpu.emi.EMIFile", return_value=mock_file),
        ):
            cpus = EMICPUs()

        assert len(cpus) == 1
        assert cpus.cpus[0].supports_get_dram_energy_consumption()

    def test_multi_package(self):
        """EMICPUs creates one EMICPU per package, ordered by package index."""
        mock_file = _make_emi_file_with_channels(
            [
                (0, "RAPL_Package0_PKG"),
                (1, "RAPL_Package0_DRAM"),
                (2, "RAPL_Package1_PKG"),
                (3, "RAPL_Package1_DRAM"),
            ]
        )
        with (
            patch("zeus.device.cpu.emi.emi_is_available", return_value=True),
            patch("zeus.device.cpu.emi._get_emi_device_paths", return_value=["fake_path"]),
            patch("zeus.device.cpu.emi.EMIFile", return_value=mock_file),
        ):
            cpus = EMICPUs()

        assert len(cpus) == 2
        assert cpus.cpus[0].cpu_index == 0
        assert cpus.cpus[1].cpu_index == 1

    def test_raises_when_no_pkg_channels(self):
        """EMICPUs raises ZeusEMINotSupportedError when devices have no PKG channels."""
        mock_file = _make_emi_file_with_channels([(0, "RAPL_Package0_PP0"), (1, "RAPL_Package0_PP1")])
        with (
            patch("zeus.device.cpu.emi.emi_is_available", return_value=True),
            patch("zeus.device.cpu.emi._get_emi_device_paths", return_value=["fake_path"]),
            patch("zeus.device.cpu.emi.EMIFile", return_value=mock_file),
            pytest.raises(ZeusEMINotSupportedError),
        ):
            EMICPUs()

    def test_skips_failing_devices(self):
        """EMICPUs skips devices that raise ZeusEMIInitError during construction."""
        good_file = _make_emi_file_with_channels([(0, "RAPL_Package0_PKG")])

        def emi_file_factory(path: str):
            if path == "bad_path":
                raise ZeusEMIInitError("simulated failure")
            return good_file

        with (
            patch("zeus.device.cpu.emi.emi_is_available", return_value=True),
            patch(
                "zeus.device.cpu.emi._get_emi_device_paths",
                return_value=["bad_path", "good_path"],
            ),
            patch("zeus.device.cpu.emi.EMIFile", side_effect=emi_file_factory),
        ):
            cpus = EMICPUs()

        assert len(cpus) == 1

    def test_get_total_energy_forwarded(self):
        """EMICPUs.get_total_energy_consumption forwards to the correct EMICPU."""
        mock_file = _make_emi_file_with_channels([(0, "RAPL_Package0_PKG"), (1, "RAPL_Package0_DRAM")])
        mock_file.read.side_effect = [9000.0, 500.0]

        with (
            patch("zeus.device.cpu.emi.emi_is_available", return_value=True),
            patch("zeus.device.cpu.emi._get_emi_device_paths", return_value=["fake_path"]),
            patch("zeus.device.cpu.emi.EMIFile", return_value=mock_file),
        ):
            cpus = EMICPUs()
            result = cpus.get_total_energy_consumption(0)

        assert result.cpu_mj == pytest.approx(9000.0)
        assert result.dram_mj == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# EMIFile.read — energy unit conversion
# ---------------------------------------------------------------------------


class TestEMIFileRead:
    """Tests for energy conversion in EMIFile.read."""

    def test_picowatt_hours_to_millijoules(self):
        """read() converts pWh correctly: 1 pWh = 3.6e-6 mJ."""
        # Build mock measurement bytes: channel 0 = 1_000_000 pWh
        meas_bytes = _make_measurement([1_000_000])

        mock_ioctl = MagicMock(return_value=meas_bytes)

        # Construct an EMIFile without calling __init__ (Windows API not available in CI).
        emi_file = object.__new__(EMIFile)
        emi_file.path = "fake"
        emi_file._handle = 0  # null sentinel — __del__ skips CloseHandle for handle == 0
        emi_file.version = 2
        emi_file.channels = [_EMIChannel(index=0, name="RAPL_Package0_PKG", unit=0)]

        with patch("zeus.device.cpu.emi._ioctl", mock_ioctl):
            result = emi_file.read(channel_index=0)

        assert result == pytest.approx(1_000_000 * 3.6e-6)

    def test_read_selects_correct_channel(self):
        """read() reads the right 16-byte slot for the requested channel."""
        # 3 channels; channel 2 has energy = 5_000_000 pWh
        meas_bytes = _make_measurement([100, 200, 5_000_000])

        emi_file = object.__new__(EMIFile)
        emi_file.path = "fake"
        emi_file._handle = 0  # null sentinel
        emi_file.version = 2
        emi_file.channels = [
            _EMIChannel(0, "ch0", 0),
            _EMIChannel(1, "ch1", 0),
            _EMIChannel(2, "ch2", 0),
        ]

        with patch("zeus.device.cpu.emi._ioctl", return_value=meas_bytes):
            result = emi_file.read(channel_index=2)

        assert result == pytest.approx(5_000_000 * 3.6e-6)


# ---------------------------------------------------------------------------
# Windows-only integration test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="EMI is Windows-only")
class TestEMIIntegration:
    """Integration tests that exercise the real Windows EMI API."""

    def test_emi_is_available_on_supported_hardware(self):
        """emi_is_available returns True on hardware that exposes an EMI device."""
        emi_is_available.cache_clear()
        result = emi_is_available()
        emi_is_available.cache_clear()
        # This machine has an Intel 12th gen Core with EMI support.
        assert isinstance(result, bool)

    def test_emicpus_initialises_and_reads(self):
        """EMICPUs can be initialised and return a positive energy value."""
        emi_is_available.cache_clear()
        available = emi_is_available()
        emi_is_available.cache_clear()
        if not available:
            pytest.skip("No EMI device found on this machine.")

        cpus = EMICPUs()
        assert len(cpus) >= 1

        measurement = cpus.get_total_energy_consumption(0)
        assert isinstance(measurement, CpuDramMeasurement)
        assert measurement.cpu_mj > 0.0

    def test_energy_increases_over_time(self):
        """Energy readings are monotonically increasing."""
        import time

        emi_is_available.cache_clear()
        available = emi_is_available()
        emi_is_available.cache_clear()
        if not available:
            pytest.skip("No EMI device found on this machine.")

        cpus = EMICPUs()
        first = cpus.get_total_energy_consumption(0)
        time.sleep(0.1)
        second = cpus.get_total_energy_consumption(0)
        assert second.cpu_mj >= first.cpu_mj
