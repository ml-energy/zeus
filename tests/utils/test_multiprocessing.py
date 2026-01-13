"""Tests for zeus.utils.multiprocessing module."""

from __future__ import annotations

import multiprocessing
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest
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

    def test_no_warning_when_not_at_module_level(self, mocker: MockerFixture):
        """No warning if not at module level (e.g., inside a function)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            monitor = FakeMonitor()
            assert len(w) == 0

    def test_warning_when_global_in_spawned_child(self, mocker: MockerFixture):
        """Warning should be raised when in spawned child at module level."""
        # Mock the detection function directly
        mocker.patch(
            "zeus.utils.multiprocessing._is_global_in_spawned_child",
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

    def test_warning_is_raised_for_global_monitor_in_subprocess(self, tmp_path):
        """Integration test: warning is raised for a global monitor in a spawned subprocess."""
        # The root of the project, to be added to PYTHONPATH for the subprocess
        project_root = Path(__file__).parent.parent.parent

        script_content = textwrap.dedent(f'''
            import sys
            import multiprocessing

            # Add project root to path to allow importing from tests and zeus
            sys.path.insert(0, r"{project_root}")

            from tests.utils.test_multiprocessing import FakeMonitor

            # Global monitor instance. This should trigger a warning in the spawned child process.
            monitor = FakeMonitor()

            def dummy_worker():
                pass

            if __name__ == "__main__":
                ctx = multiprocessing.get_context("spawn")
                with ctx.Pool(1) as pool:
                    pool.apply(dummy_worker)
        ''')

        script_path = tmp_path / "global_monitor_test.py"
        script_path.write_text(script_content)

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert "FakeMonitor was instantiated during module import" in result.stderr
        assert "spawned subprocess" in result.stderr
