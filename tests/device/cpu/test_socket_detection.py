from __future__ import annotations

import builtins
import os
import pytest
import platform

from unittest.mock import patch

from zeus.device.cpu.rapl import (
    RAPL_DIR,
    CONTAINER_RAPL_DIR,
    rapl_is_available,
    get_current_rapl_zone_id,
)


class MockFile:
    def __init__(self, content):
        self.content = content

    def read(self, *args, **kwargs):
        return self.content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


@pytest.fixture
def mock_linux_files():
    STAT_CONTENT = (
        "6511 (cat) R 6501 6511 6501 34816 6511 4194304 95 0 0 0 0 "
        "0 0 0 20 0 1 0 7155 6053888 255 18446744073709551615 94288847015936 "
        "94288847031350 140727519907328 0 0 0 0 0 0 0 0 0 17 24 0 0 0 0 0 "
        "94288847043296 94288847044712 94288866566144 140727519913901 "
        "140727519913921 140727519913921 140727519916011 0"
    )
    # The 39th field (index 38) of the stat content above is the logical CPU, 24.
    files = {
        "/proc/515/stat": STAT_CONTENT,
        "/sys/devices/system/cpu/cpu24/topology/physical_package_id": "1",
        "/sys/devices/system/cpu/cpu24/topology/die_id": "0",
        f"{RAPL_DIR}:0/name": "package-0",
        f"{RAPL_DIR}:1/name": "package-1",
    }

    real_open = builtins.open

    def mock_file_open(filepath, *args, **kwargs):
        if filepath in files:
            return MockFile(files[filepath])
        return real_open(filepath, *args, **kwargs)

    def mock_glob(pattern):
        # Emulate globbing the top-level RAPL package zones.
        return [f"{RAPL_DIR}:0", f"{RAPL_DIR}:1"]

    with (
        patch("builtins.open", side_effect=mock_file_open),
        patch("zeus.device.cpu.rapl.glob", side_effect=mock_glob),
        patch("os.path.exists", return_value=True),
    ):
        yield


def test_get_current_rapl_zone_id(mock_linux_files):
    # The process runs on package 1, whose RAPL zone (name "package-1") is zone 1.
    assert get_current_rapl_zone_id(515) == 1


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux specific test")
def test_get_current_rapl_zone_id_linux():
    assert platform.system() == "Linux"
    if not rapl_is_available():
        pytest.skip("RAPL is not available on this machine")

    zone_id = get_current_rapl_zone_id()
    # The returned zone must be a real top-level RAPL package zone.
    rapl_dir = RAPL_DIR if os.path.exists(RAPL_DIR) else CONTAINER_RAPL_DIR
    assert os.path.exists(f"{rapl_dir}:{zone_id}/name")
