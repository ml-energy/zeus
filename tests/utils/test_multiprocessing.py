"""Tests for zeus.utils.multiprocessing module."""

from __future__ import annotations

import multiprocessing
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

    def test_no_warning_when_called_from_function(self):
        """Even in a subprocess, no warning if instantiated from inside a function."""
        # Mock parent_process to simulate being in a subprocess
        mock_parent = mock.MagicMock()
        with mock.patch(
            "zeus.utils.multiprocessing.multiprocessing.parent_process",
            return_value=mock_parent,
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # We're instantiating from inside a test function, so no warning
                monitor = FakeMonitor()
                assert len(w) == 0

    def test_warning_when_module_level_in_subprocess(self):
        """Warning should be raised when at module level in a subprocess."""
        # Mock parent_process to simulate being in a subprocess
        mock_parent = mock.MagicMock()
        # Mock the module-level detection to return True
        with mock.patch(
            "zeus.utils.multiprocessing.multiprocessing.parent_process",
            return_value=mock_parent,
        ):
            with mock.patch(
                "zeus.utils.multiprocessing._is_being_called_at_module_level",
                return_value=True,
            ):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    warn_if_global_in_subprocess("TestClass")
                    assert len(w) == 1
                    assert "TestClass" in str(w[0].message)
                    assert "module level" in str(w[0].message)
                    assert "if __name__ == '__main__':" in str(w[0].message)


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
