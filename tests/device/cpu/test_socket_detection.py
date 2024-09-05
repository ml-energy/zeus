from __future__ import annotations

import builtins
import os
import pytest
from unittest.mock import patch

from zeus.device.cpu import get_current_cpu_index

class MockFile:
  def __init__(self, file_path):
    self.file_path = file_path

  def read(self, *args, **kwargs):
    with open(self.file_path, "r") as file:
      return file.read()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    pass


@pytest.fixture
def mock_linux_files():
  files_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../files"))
  mocked_stat_file_path = os.path.join(files_folder, "stat")
  mocked_phys_package_path = os.path.join(files_folder, "physical_package_id")
  mocked_stat_file = MockFile(mocked_stat_file_path)
  mocked_phys_package_file = MockFile(mocked_phys_package_path)

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
  assert(get_current_cpu_index(515) == 0)