"""Tests for zeus.utils.multiprocessing module."""

from __future__ import annotations

import multiprocessing
import sys
import warnings
from unittest import mock

import pytest

from zeus.utils.multiprocessing import warn_if_global_in_subprocess


class FakeMonitor:
    """A class that mimics the ZeusMonitor pattern for testing."""

    def __init__(self):
        warn_if_global_in_subprocess("FakeMonitor")


class TestWarnIfGlobalInSubprocess:
    """Test warn_if_global_in_subprocess function."""

    def test_no_warning_in_main_process(self):
        """In the main process, no warning should be raised."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Instantiate through a class like ZeusMonitor does
            monitor = FakeMonitor()
            assert len(w) == 0

    def test_no_warning_when_not_spawned_child(self):
        """No warning if not in a spawned child process."""
        # Even with __mp_main__ in sys.modules, if parent_process is None, no warning
        with mock.patch.dict(sys.modules, {"__mp_main__": mock.MagicMock()}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                monitor = FakeMonitor()
                assert len(w) == 0

    def test_warning_when_spawned_child_at_module_level(self):
        """Warning should be raised when in spawned child at module level."""
        mock_parent = mock.MagicMock()
        with mock.patch(
            "zeus.utils.multiprocessing.mp.parent_process",
            return_value=mock_parent,
        ):
            with mock.patch.dict(sys.modules, {"__mp_main__": mock.MagicMock()}):
                with mock.patch(
                    "zeus.utils.multiprocessing._called_from_module_level",
                    return_value=True,
                ):
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        warn_if_global_in_subprocess("TestClass")
                        assert len(w) == 1
                        assert "TestClass" in str(w[0].message)
                        assert "module import" in str(w[0].message)
                        assert '__name__ == "__main__"' in str(w[0].message)


class TestIntegration:
    """Integration tests that verify the detection logic end-to-end."""

    def test_subprocess_no_warning_when_instantiated_in_function(self):
        """Instantiating in a subprocess inside a function should not warn."""

        def worker_function():
            """Function that runs in subprocess and instantiates FakeMonitor."""
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # Instantiate inside a function - should NOT warn
                monitor = FakeMonitor()
                return len(w)

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            result = pool.apply(worker_function)
            # Should be 0 warnings
            assert result == 0

    def test_main_process_detection(self):
        """Verify that parent_process() returns None in main process."""
        # This is mostly a sanity check
        assert multiprocessing.parent_process() is None

    def test_main_process_instantiation_no_warning(self):
        """Instantiating in main process should never warn, even at module level."""
        # This test verifies that in the main process, we never warn
        # regardless of where the instantiation happens
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            monitor = FakeMonitor()
            assert len(w) == 0


class TestMonitorIntegration:
    """Test that each monitor class calls warn_if_global_in_subprocess."""

    def test_zeus_monitor_calls_warning(self):
        """ZeusMonitor should call warn_if_global_in_subprocess with correct name."""
        with mock.patch("zeus.monitor.energy.warn_if_global_in_subprocess") as mock_warn:
            # Mock dependencies to avoid actual GPU/CPU initialization
            with mock.patch("zeus.monitor.energy.get_gpus") as mock_gpus:
                with mock.patch("zeus.monitor.energy.get_cpus") as mock_cpus:
                    with mock.patch("zeus.monitor.energy.get_soc") as mock_soc:
                        mock_gpus.return_value = mock.MagicMock()
                        mock_gpus.return_value.__len__ = mock.MagicMock(return_value=0)
                        mock_cpus.return_value = mock.MagicMock()
                        mock_cpus.return_value.cpus = []
                        mock_soc.return_value = mock.MagicMock()
                        mock_soc.return_value.__len__ = mock.MagicMock(return_value=0)

                        from zeus.monitor.energy import ZeusMonitor

                        try:
                            monitor = ZeusMonitor()
                        except Exception:
                            pass  # We only care that the warning function was called

                        mock_warn.assert_called_once_with("ZeusMonitor")

    def test_power_monitor_calls_warning(self):
        """PowerMonitor should call warn_if_global_in_subprocess with correct name."""
        with mock.patch("zeus.monitor.power.warn_if_global_in_subprocess") as mock_warn:
            with mock.patch("zeus.monitor.power.get_gpus") as mock_gpus:
                mock_gpu_obj = mock.MagicMock()
                mock_gpu_obj.__len__ = mock.MagicMock(return_value=1)
                mock_gpus.return_value = mock_gpu_obj

                from zeus.monitor.power import PowerMonitor

                try:
                    monitor = PowerMonitor(gpu_indices=[0])
                except Exception:
                    pass  # We only care that the warning function was called

                mock_warn.assert_called_once_with("PowerMonitor")

    def test_temperature_monitor_calls_warning(self):
        """TemperatureMonitor should call warn_if_global_in_subprocess with correct name."""
        with mock.patch("zeus.monitor.temperature.warn_if_global_in_subprocess") as mock_warn:
            with mock.patch("zeus.monitor.temperature.get_gpus") as mock_gpus:
                mock_gpu_obj = mock.MagicMock()
                mock_gpu_obj.__len__ = mock.MagicMock(return_value=1)
                mock_gpus.return_value = mock_gpu_obj

                from zeus.monitor.temperature import TemperatureMonitor

                try:
                    monitor = TemperatureMonitor(gpu_indices=[0])
                except Exception:
                    pass  # We only care that the warning function was called

                mock_warn.assert_called_once_with("TemperatureMonitor")

    def test_carbon_emission_monitor_calls_warning(self):
        """CarbonEmissionMonitor should call warn_if_global_in_subprocess with correct name."""
        with mock.patch("zeus.monitor.carbon.warn_if_global_in_subprocess") as mock_warn:
            with mock.patch("zeus.monitor.carbon.ZeusMonitor"):
                from zeus.monitor.carbon import CarbonEmissionMonitor

                mock_provider = mock.MagicMock()
                try:
                    monitor = CarbonEmissionMonitor(carbon_intensity_provider=mock_provider)
                except Exception:
                    pass  # We only care that the warning function was called

                mock_warn.assert_called_once_with("CarbonEmissionMonitor")

    def test_energy_cost_monitor_calls_warning(self):
        """EnergyCostMonitor should call warn_if_global_in_subprocess with correct name."""
        with mock.patch("zeus.monitor.price.warn_if_global_in_subprocess") as mock_warn:
            with mock.patch("zeus.monitor.price.ZeusMonitor"):
                from zeus.monitor.price import EnergyCostMonitor

                mock_provider = mock.MagicMock()
                try:
                    monitor = EnergyCostMonitor(electricity_price_provider=mock_provider)
                except Exception:
                    pass  # We only care that the warning function was called

                mock_warn.assert_called_once_with("EnergyCostMonitor")
