"""Tests for zeus.utils.multiprocessing module."""

from __future__ import annotations

import multiprocessing
import sys
import warnings
from unittest.mock import MagicMock

from pytest_mock import MockerFixture

from zeus.utils.multiprocessing import warn_if_global_in_subprocess


class FakeMonitor:
    """A class that mimics the ZeusMonitor pattern for testing."""

    def __init__(self):
        warn_if_global_in_subprocess(self)


# Module-level function for subprocess test (local functions can't be pickled)
def _worker_function_for_subprocess_test():
    """Function that runs in subprocess and instantiates FakeMonitor."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        monitor = FakeMonitor()
        return len(w)


class TestWarnIfGlobalInSubprocess:
    """Test warn_if_global_in_subprocess function."""

    def test_no_warning_in_main_process(self):
        """In the main process, no warning should be raised."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            monitor = FakeMonitor()
            assert len(w) == 0

    def test_no_warning_when_not_spawned_child(self, mocker: MockerFixture):
        """No warning if not in a spawned child process."""
        # Even with __mp_main__ in sys.modules, if parent_process is None, no warning
        mocker.patch.dict(sys.modules, {"__mp_main__": MagicMock()})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            monitor = FakeMonitor()
            assert len(w) == 0

    def test_warning_when_spawned_child_at_module_level(self, mocker: MockerFixture):
        """Warning should be raised when in spawned child at module level."""
        mocker.patch(
            "zeus.utils.multiprocessing.mp.parent_process",
            return_value=MagicMock(),
        )
        mocker.patch.dict(sys.modules, {"__mp_main__": MagicMock()})
        mocker.patch(
            "zeus.utils.multiprocessing._called_from_module_level",
            return_value=True,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Pass an object so type(self).__name__ works
            warn_if_global_in_subprocess(FakeMonitor.__new__(FakeMonitor))
            assert len(w) == 1
            assert "FakeMonitor" in str(w[0].message)
            assert "module import" in str(w[0].message)
            assert '__name__ == "__main__"' in str(w[0].message)


class TestIntegration:
    """Integration tests that verify the detection logic end-to-end."""

    def test_subprocess_no_warning_when_instantiated_in_function(self):
        """Instantiating in a subprocess inside a function should not warn."""
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            result = pool.apply(_worker_function_for_subprocess_test)
            assert result == 0

    def test_main_process_detection(self):
        """Verify that parent_process() returns None in main process."""
        assert multiprocessing.parent_process() is None

    def test_main_process_instantiation_no_warning(self):
        """Instantiating in main process should never warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            monitor = FakeMonitor()
            assert len(w) == 0
