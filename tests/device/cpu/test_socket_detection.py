from __future__ import annotations

import builtins
import os
import pytest
import platform

from unittest.mock import patch

from zeus.device.cpu import get_current_cpu_index


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
    PHYSICAL_PACKAGE_CONTENT = "0"

    mocked_stat_file = MockFile(STAT_CONTENT)
    mocked_phys_package_file = MockFile(PHYSICAL_PACKAGE_CONTENT)

    real_open = builtins.open

    def mock_file_open(filepath, *args, **kwargs):
        if filepath == "/proc/515/stat":
            return mocked_stat_file
        elif filepath == "/sys/devices/system/cpu/cpu24/topology/physical_package_id":
            return mocked_phys_package_file
        else:
            return real_open(filepath, *args, **kwargs)

    patch_open = patch("builtins.open", side_effect=mock_file_open)

    patch_open.start()

    yield

    patch_open.stop()


def test_get_current_cpu_index(mock_linux_files):
    assert get_current_cpu_index(515) == 0


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux specific test")
def test_get_current_cpu_index_linux():
    assert platform.system() == "Linux"
    assert os.path.exists("/proc/cpuinfo")
    socket_ids = set()
    with open("/proc/cpuinfo") as f:
        for line in f:
            if "physical id" in line:
                socket_id = line.strip().split(":")[1].strip()
                socket_ids.add(socket_id)

    cpu_index = get_current_cpu_index()
    assert cpu_index >= 0 and cpu_index < len(socket_ids)
