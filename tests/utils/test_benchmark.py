"""Tests for zeus.utils.benchmark module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from zeus.monitor import Measurement
from zeus.utils.benchmark import measure


class TestMeasure:
    """Tests for the measure function."""

    def test_measure_basic(self):
        """Test basic functionality with a mock monitor."""
        # Create a mock ZeusMonitor
        mock_monitor = MagicMock()
        mock_measurement = Measurement(
            time=1.5,
            gpu_energy={0: 10.0},
        )
        mock_monitor.end_window.return_value = mock_measurement

        # Create a simple function to measure
        call_count = 0

        def simple_func():
            nonlocal call_count
            call_count += 1

        # Measure the function
        result = measure(simple_func, zeus_monitor=mock_monitor)

        # Verify the function was called once
        assert call_count == 1

        # Verify monitor methods were called
        mock_monitor.begin_window.assert_called_once_with("__zeus_measure_window__")
        mock_monitor.end_window.assert_called_once_with("__zeus_measure_window__")

        # Verify the result
        assert result == mock_measurement
        assert result.time == 1.5
        assert result.gpu_energy == {0: 10.0}

    def test_measure_with_args(self):
        """Test measure with positional arguments."""
        mock_monitor = MagicMock()
        mock_monitor.end_window.return_value = Measurement(time=1.0, gpu_energy={0: 5.0})

        received_args = []

        def func_with_args(a, b, c):
            received_args.extend([a, b, c])

        measure(func_with_args, args=[1, 2, 3], zeus_monitor=mock_monitor)

        assert received_args == [1, 2, 3]

    def test_measure_with_kwargs(self):
        """Test measure with keyword arguments."""
        mock_monitor = MagicMock()
        mock_monitor.end_window.return_value = Measurement(time=1.0, gpu_energy={0: 5.0})

        received_kwargs = {}

        def func_with_kwargs(**kwargs):
            received_kwargs.update(kwargs)

        measure(func_with_kwargs, kwargs={"x": 10, "y": 20}, zeus_monitor=mock_monitor)

        assert received_kwargs == {"x": 10, "y": 20}

    def test_measure_with_args_and_kwargs(self):
        """Test measure with both positional and keyword arguments."""
        mock_monitor = MagicMock()
        mock_monitor.end_window.return_value = Measurement(time=1.0, gpu_energy={0: 5.0})

        received = {}

        def func_with_both(a, b, c=None, d=None):
            received["args"] = (a, b)
            received["kwargs"] = {"c": c, "d": d}

        measure(
            func_with_both,
            args=["arg1", "arg2"],
            kwargs={"c": "kwarg1", "d": "kwarg2"},
            zeus_monitor=mock_monitor,
        )

        assert received["args"] == ("arg1", "arg2")
        assert received["kwargs"] == {"c": "kwarg1", "d": "kwarg2"}

    def test_measure_num_repeats(self):
        """Test measure with multiple repetitions."""
        mock_monitor = MagicMock()
        mock_monitor.end_window.return_value = Measurement(time=5.0, gpu_energy={0: 50.0})

        call_count = 0

        def counting_func():
            nonlocal call_count
            call_count += 1

        measure(counting_func, num_repeats=5, zeus_monitor=mock_monitor)

        # The function should be called 5 times
        assert call_count == 5

        # But begin_window and end_window should only be called once each
        mock_monitor.begin_window.assert_called_once()
        mock_monitor.end_window.assert_called_once()

    def test_measure_invalid_num_repeats(self):
        """Test measure raises ValueError for invalid num_repeats."""
        mock_monitor = MagicMock()

        with pytest.raises(ValueError, match="num_repeats must be at least 1"):
            measure(lambda: None, num_repeats=0, zeus_monitor=mock_monitor)

        with pytest.raises(ValueError, match="num_repeats must be at least 1"):
            measure(lambda: None, num_repeats=-1, zeus_monitor=mock_monitor)

    def test_measure_zero_energy_warning(self):
        """Test measure warns when energy consumption is zero."""
        mock_monitor = MagicMock()
        mock_monitor.end_window.return_value = Measurement(
            time=0.001,
            gpu_energy={0: 0.0},  # Zero energy
        )

        with pytest.warns(
            UserWarning,
            match="The energy consumption was measured as zero",
        ):
            measure(lambda: None, zeus_monitor=mock_monitor)

    def test_measure_multi_gpu_zero_energy_warning(self):
        """Test measure warns when any GPU has zero energy."""
        mock_monitor = MagicMock()
        mock_monitor.end_window.return_value = Measurement(
            time=0.5,
            gpu_energy={0: 10.0, 1: 0.0, 2: 15.0},  # One GPU with zero energy
        )

        with pytest.warns(
            UserWarning,
            match="The energy consumption was measured as zero",
        ):
            measure(lambda: None, zeus_monitor=mock_monitor)

    def test_measure_no_warning_with_energy(self):
        """Test measure does not warn when energy is non-zero."""
        mock_monitor = MagicMock()
        mock_monitor.end_window.return_value = Measurement(
            time=1.0,
            gpu_energy={0: 10.0, 1: 15.0},
        )

        # Should not raise any warnings
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            measure(lambda: None, zeus_monitor=mock_monitor)

    def test_measure_creates_monitor_if_not_provided(self):
        """Test measure creates a ZeusMonitor if not provided."""
        with patch("zeus.monitor.energy.ZeusMonitor") as mock_monitor_class:
            mock_instance = MagicMock()
            mock_instance.end_window.return_value = Measurement(
                time=1.0,
                gpu_energy={0: 10.0},
            )
            mock_monitor_class.return_value = mock_instance

            measure(lambda: None)

            # Verify a monitor was created
            mock_monitor_class.assert_called_once()

    def test_measure_returns_measurement(self):
        """Test measure returns the correct Measurement object."""
        mock_monitor = MagicMock()
        expected_measurement = Measurement(
            time=2.5,
            gpu_energy={0: 25.0, 1: 30.0},
            cpu_energy={0: 5.0},
            dram_energy={0: 1.0},
        )
        mock_monitor.end_window.return_value = expected_measurement

        result = measure(lambda: None, zeus_monitor=mock_monitor)

        assert result is expected_measurement
        assert result.time == 2.5
        assert result.gpu_energy == {0: 25.0, 1: 30.0}
        assert result.cpu_energy == {0: 5.0}
        assert result.dram_energy == {0: 1.0}

    def test_measure_with_empty_args_kwargs(self):
        """Test measure with explicitly empty args and kwargs."""
        mock_monitor = MagicMock()
        mock_monitor.end_window.return_value = Measurement(time=1.0, gpu_energy={0: 5.0})

        received = {"args": None, "kwargs": None}

        def capture_func(*args, **kwargs):
            received["args"] = args
            received["kwargs"] = kwargs

        measure(capture_func, args=[], kwargs={}, zeus_monitor=mock_monitor)

        assert received["args"] == ()
        assert received["kwargs"] == {}

    def test_measure_with_none_args_kwargs(self):
        """Test measure with None args and kwargs (default behavior)."""
        mock_monitor = MagicMock()
        mock_monitor.end_window.return_value = Measurement(time=1.0, gpu_energy={0: 5.0})

        received = {"args": None, "kwargs": None}

        def capture_func(*args, **kwargs):
            received["args"] = args
            received["kwargs"] = kwargs

        measure(capture_func, args=None, kwargs=None, zeus_monitor=mock_monitor)

        assert received["args"] == ()
        assert received["kwargs"] == {}
