# Copyright (C) 2023 Jae-Won Chung <jwnchung@umich.edu>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import itertools
from unittest.mock import call
from itertools import product, combinations
from typing import Generator, TYPE_CHECKING, Sequence

import pynvml
import pytest

from zeus.monitor import Measurement, ZeusMonitor
from zeus.util.testing import ReplayZeusMonitor

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock
    from pytest_mock import MockerFixture

NUM_GPUS = 4
ARCHS = [
    pynvml.NVML_DEVICE_ARCH_PASCAL,
    pynvml.NVML_DEVICE_ARCH_VOLTA,
    pynvml.NVML_DEVICE_ARCH_AMPERE,
]


@pytest.fixture
def pynvml_mock(mocker: MockerFixture):
    """Mock the entire pynvml module."""
    mock = mocker.patch("zeus.monitor.energy.pynvml", autospec=True)
    
    # Except for the arch constants.
    mock.NVML_DEVICE_ARCH_PASCAL = pynvml.NVML_DEVICE_ARCH_PASCAL
    mock.NVML_DEVICE_ARCH_VOLTA = pynvml.NVML_DEVICE_ARCH_VOLTA
    mock.NVML_DEVICE_ARCH_AMPERE = pynvml.NVML_DEVICE_ARCH_AMPERE

    mocker.patch("zeus.util.env.pynvml", mock)

    return mock


def gpu_cases():
    """Generates all the cases for different GPU indices and architectures.

    We start with `NUM_GPUS` GPUs. We then consider all possible combinations of
    `CUDA_VISIBLE_DEVICES`, which limits the set of GPUs visible to PyTorch and
    creates a mapping between the indices visible to PyTorch and the indices visible
    to NVML. We then consider all possible combinations of `gpu_indices`, which
    creates a mapping between the indices visible to PyTorch and the indices visible
    to `ZeusMonitor`. Finally, with only the GPUs that were left after filtering
    with `CUDA_VISIBLE_DEVICES` and `gpu_monitor`, we assign all possible combinations
    GPU architectures to `gpu_archs`.
    """
    def nonempty_subsequences(indices: Sequence[int]) -> Generator[list[int], None, None]:
        """Returns all non-empty subsets of `indices`."""
        for length in range(1, len(indices) + 1):
            yield from map(list, combinations(indices, r=length))

    # `CUDA_VISIBLE_DEVICES` is given.
    for cuda_visible_devices in nonempty_subsequences(range(NUM_GPUS)):  # NVML index
        # `gpu_indices` is not `None`.
        for gpu_indices in nonempty_subsequences(range(len(cuda_visible_devices))):  # PyTorch index
            for gpu_archs in product(ARCHS, repeat=len(gpu_indices)):
                yield cuda_visible_devices, gpu_indices, list(gpu_archs)
        # `gpu_indices` is `None` (use all GPUs visible to PyTorch).
        for gpu_archs in product(ARCHS, repeat=len(cuda_visible_devices)):
            yield cuda_visible_devices, None, list(gpu_archs)
    # `CUDA_VISIBLE_DEVICES` is not given (use all GPUs in the system).
    # `gpu_indices` is not `None`.
    for gpu_indices in nonempty_subsequences(range(NUM_GPUS)):  # PyTorch index
        for gpu_archs in product(ARCHS, repeat=len(gpu_indices)):
            yield None, gpu_indices, list(gpu_archs)
    # `gpu_indices` is `None` (use all GPUs visible to PyTorch).
    for gpu_archs in product(ARCHS, repeat=NUM_GPUS):
        yield None, None, list(gpu_archs)


@pytest.fixture(params=gpu_cases())
def mock_gpus(request, mocker: MockerFixture, pynvml_mock: MagicMock) -> tuple[tuple[int], tuple[int]]:
    """Mock `pynvml` so that it looks like there are GPUs with the given archs.

    This fixture automatically generates (1) all combinations of possible GPU indices
    (e.g., (0,), (1, 3), (0, 2, 3)) and takes assigns all possible combinations of
    GPU architectures (e.g., (PASCAL,), (VOLTA, VOLTA), (AMPERE, PASCAL, VOLTA)), and
    (2) all combinations of selecting four GPU architectures with replacement and
    matches it with `None` for the GPU indices (e.g., (None, (PASCAL, PASCAL, VOLTA, AMPERE))).
    The latter case is to test the initialization of `ZeusMonitor` without any input
    argument.

    Thus `request.param` has type `tuple[tuple[int], tuple[int]]` where the former
    is GPU indices and the latter is GPU architectures.
    """
    cuda_visible_devices, gpu_indices, archs = request.param

    def mock_pynvml(nvml_indices: list[int], archs: list[int]) -> None:
        assert len(nvml_indices) == len(archs)
        index_to_handle = {i: f"handle{i}" for i in nvml_indices}
        handle_to_arch = {f"handle{i}": arch for i, arch in zip(nvml_indices, archs)}
        pynvml_mock.nvmlDeviceGetCount.return_value = NUM_GPUS
        pynvml_mock.nvmlDeviceGetHandleByIndex.side_effect = lambda index: index_to_handle[index]
        pynvml_mock.nvmlDeviceGetArchitecture.side_effect = lambda handle: handle_to_arch[handle]

    if cuda_visible_devices is None:  # All GPUs are visible to PyTorch.
        # When `CUDA_VISIBLE_DEVICES` is not given, NVML indices and PyTorch indices conincide.
        if gpu_indices is None:
            mock_pynvml(list(range(NUM_GPUS)), archs)
        else:
            mock_pynvml(gpu_indices, archs)

    else:  # `CUDA_VISIBLE_DEVICES` is given, limiting the set of visible GPUs to PyTorch.
        mocker.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ",".join(map(str, cuda_visible_devices))})
        # Valid NVML indices are determined by `cuda_visible_devices` and `gpu_indices`.
        if gpu_indices is None:
            mock_pynvml(cuda_visible_devices, archs)
        else:
            # We need to translate `gpu_indices` to NVML indices.
            mock_pynvml([cuda_visible_devices[idx] for idx in gpu_indices], archs)

    return request.param


def test_monitor(pynvml_mock, mock_gpus, mocker: MockerFixture, tmp_path: Path):
    """Test the `ZeusMonitor` class."""
    cuda_visible_devices, gpu_indices, gpu_archs = mock_gpus
    if cuda_visible_devices is None:
        if gpu_indices is None:
            nvml_gpu_indices = torch_gpu_indices = list(range(NUM_GPUS))
        else:
            nvml_gpu_indices = torch_gpu_indices = gpu_indices
    else:
        if gpu_indices is None:
            nvml_gpu_indices = cuda_visible_devices
            torch_gpu_indices = list(range(len(cuda_visible_devices)))
        else:
            nvml_gpu_indices = [cuda_visible_devices[idx] for idx in gpu_indices]
            torch_gpu_indices = gpu_indices
    assert len(nvml_gpu_indices) == len(torch_gpu_indices) == len(gpu_archs)

    num_gpus = len(gpu_archs)
    is_old_nvml = {index: arch < pynvml.NVML_DEVICE_ARCH_VOLTA for index, arch in zip(nvml_gpu_indices, gpu_archs)}
    is_old_torch = {index: arch < pynvml.NVML_DEVICE_ARCH_VOLTA for index, arch in zip(torch_gpu_indices, gpu_archs)}
    old_gpu_torch_indices = [index for index, is_old in is_old_torch.items() if is_old]

    mocker.patch("zeus.monitor.energy.atexit.register")

    class MockPowerMonitor:
        def __init__(self, gpu_indices: list[int] | None, update_period: float | None) -> None:
            assert gpu_indices == old_gpu_torch_indices
            self.gpu_indices = gpu_indices
            self.update_period = update_period
        def get_energy(self, start: float, end: float) -> dict[int, float]:
            return {i: -1.0 for i in self.gpu_indices}
    mocker.patch("zeus.monitor.energy.PowerMonitor", MockPowerMonitor)

    time_counter = itertools.count(start=4, step=1)
    mocker.patch("zeus.monitor.energy.time", side_effect=time_counter)

    energy_counters = {
        f"handle{i}": itertools.count(start=1000, step=3)
        for i in nvml_gpu_indices if not is_old_nvml[i]
    }
    pynvml_mock.nvmlDeviceGetTotalEnergyConsumption.side_effect = lambda handle: next(energy_counters[handle])

    log_file = tmp_path / "log.csv"

    ########################################
    # Test ZeusMonitor initialization.
    ########################################
    monitor = ZeusMonitor(gpu_indices=gpu_indices, log_file=log_file)

    # Check GPU index parsing from the log file.
    replay_monitor = ReplayZeusMonitor(gpu_indices=None, log_file=log_file)
    assert replay_monitor.gpu_indices == list(torch_gpu_indices)

    ########################################
    # Test measurement windows.
    ########################################
    def tick():
        """Calling this function will simulate a tick of time passing."""
        next(time_counter)
        for counter in energy_counters.values():
            next(counter)

    def assert_window_begin(name: str, begin_time: int):
        """Assert monitor measurement states right after a window begins."""
        assert monitor.measurement_states[name][0] == begin_time
        assert monitor.measurement_states[name][1] == {
            # `4` is the time origin of `time_counter`.
            i: pytest.approx((1000 + 3 * (begin_time - 4)) / 1000.0)
            for i in torch_gpu_indices if not is_old_torch[i]
        }
        pynvml_mock.nvmlDeviceGetTotalEnergyConsumption.assert_has_calls([
            call(f"handle{i}") for i in nvml_gpu_indices if not is_old_nvml[i]
        ])
        pynvml_mock.nvmlDeviceGetTotalEnergyConsumption.reset_mock()

    def assert_measurement(
        name: str,
        measurement: Measurement,
        begin_time: int,
        elapsed_time: int,
        assert_calls: bool = True,
    ):
        """Assert that energy functions are being called correctly.

        Args:
            name: The name of the measurement window.
            measurement: The Measurement object returned from `end_window`.
            begin_time: The time at which the window began.
            elapsed_time: The time elapsed when the window ended.
            assert_calls: Whether to assert calls to mock functions. (Default: `True`)
        """
        assert name not in monitor.measurement_states
        assert num_gpus == len(measurement.energy)
        assert elapsed_time == measurement.time
        assert set(measurement.energy.keys()) == set(torch_gpu_indices)
        for i in torch_gpu_indices:
            if not is_old_torch[i]:
                # The energy counter increments with step size 3.
                assert measurement.energy[i] == pytest.approx(elapsed_time * 3 / 1000.0)

        if not assert_calls:
            return

        pynvml_mock.nvmlDeviceGetTotalEnergyConsumption.assert_has_calls([
            call(f"handle{i}") for i in nvml_gpu_indices if not is_old_nvml[i]
        ])
        pynvml_mock.nvmlDeviceGetTotalEnergyConsumption.reset_mock()


    # Serial non-overlapping windows.
    monitor.begin_window("window1", sync_cuda=False)
    assert_window_begin("window1", 4)

    tick()

    # Calling `begin_window` again with the same name should raise an error.
    with pytest.raises(ValueError, match="already exists"):
        monitor.begin_window("window1", sync_cuda=False)

    measurement = monitor.end_window("window1", sync_cuda=False)
    assert_measurement("window1", measurement, begin_time=4, elapsed_time=2)

    tick(); tick()

    monitor.begin_window("window2", sync_cuda=False)
    assert_window_begin("window2", 9)

    tick(); tick(); tick()

    measurement = monitor.end_window("window2", sync_cuda=False)
    assert_measurement("window2", measurement, begin_time=9, elapsed_time=4)

    # Calling `end_window` again with the same name should raise an error.
    with pytest.raises(ValueError, match="does not exist"):
        monitor.end_window("window2", sync_cuda=False)

    # Calling `end_window` with a name that doesn't exist should raise an error.
    with pytest.raises(ValueError, match="does not exist"):
        monitor.end_window("window3", sync_cuda=False)

    # Overlapping windows.
    monitor.begin_window("window3", sync_cuda=False)
    assert_window_begin("window3", 14)

    tick()

    monitor.begin_window("window4", sync_cuda=False)
    assert_window_begin("window4", 16)

    tick(); tick();

    measurement = monitor.end_window("window3", sync_cuda=False)
    assert_measurement("window3", measurement, begin_time=14, elapsed_time=5)

    tick(); tick(); tick();

    measurement = monitor.end_window("window4", sync_cuda=False)
    assert_measurement("window4", measurement, begin_time=16, elapsed_time=7)


    # Nested windows.
    monitor.begin_window("window5", sync_cuda=False)
    assert_window_begin("window5", 24)

    monitor.begin_window("window6", sync_cuda=False)
    assert_window_begin("window6", 25)

    tick(); tick();

    measurement = monitor.end_window("window6", sync_cuda=False)
    assert_measurement("window6", measurement, begin_time=25, elapsed_time=3)

    tick(); tick(); tick();

    measurement = monitor.end_window("window5", sync_cuda=False)
    assert_measurement("window5", measurement, begin_time=24, elapsed_time=8)

    ########################################
    # Test content of `log_file`.
    ########################################
    with open(log_file, "r") as f:
        lines = f.readlines()

    # The first line should be the header.
    header = "start_time,window_name,elapsed_time"
    for gpu_index in torch_gpu_indices:
        header += f",gpu{gpu_index}_energy"
    header += "\n"
    assert lines[0] == header

    # The rest of the lines should be the measurements, one line per window.
    def assert_log_file_row(row: str, name: str, begin_time: int, elapsed_time: int):
        """Assert that a row in the log file is correct.

        Args:
            row: The row to check.
            name: The name of the measurement window.
            begin_time: The time at which the window began.
            elapsed_time: The time elapsed when the window ended.
        """
        assert row.startswith(f"{begin_time},{name},{elapsed_time}")
        pieces = row.split(",")
        for i, gpu_index in enumerate(torch_gpu_indices):
            if not is_old_torch[gpu_index]:
                assert float(pieces[3 + i]) == pytest.approx(elapsed_time * 3 / 1000.0)

    assert_log_file_row(lines[1], "window1", 4, 2)
    assert_log_file_row(lines[2], "window2", 9, 4)
    assert_log_file_row(lines[3], "window3", 14, 5)
    assert_log_file_row(lines[4], "window4", 16, 7)
    assert_log_file_row(lines[5], "window6", 25, 3)
    assert_log_file_row(lines[6], "window5", 24, 8)

    ########################################
    # Test replaying from the log file.
    ########################################
    # Currently the testing infrastructure does not support old GPU architectures that
    # require the Zeus monitor binary. So we skip this test if any of the GPUs are old.
    if any(is_old_nvml.values()):
        return

    replay_monitor.begin_window("window1", sync_cuda=False)
    
    # Calling `begin_window` again with the same name should raise an error.
    with pytest.raises(RuntimeError, match="is already ongoing"):
        replay_monitor.begin_window("window1", sync_cuda=False)

    measurement = replay_monitor.end_window("window1", sync_cuda=False)
    assert_measurement("window1", measurement, begin_time=5, elapsed_time=2, assert_calls=False)

    # Calling `end_window` with a non-existant window name should raise an error.
    with pytest.raises(RuntimeError, match="is not ongoing"):
        replay_monitor.end_window("window2", sync_cuda=False)

    replay_monitor.begin_window("window2", sync_cuda=False)
    measurement = replay_monitor.end_window("window2", sync_cuda=False)
    assert_measurement("window2", measurement, begin_time=10, elapsed_time=4, assert_calls=False)

    replay_monitor.begin_window("window3", sync_cuda=False)
    replay_monitor.begin_window("window4", sync_cuda=False)

    measurement = replay_monitor.end_window("window3", sync_cuda=False)
    assert_measurement("window3", measurement, begin_time=15, elapsed_time=5, assert_calls=False)
    measurement = replay_monitor.end_window("window4", sync_cuda=False)
    assert_measurement("window4", measurement, begin_time=17, elapsed_time=7, assert_calls=False)

    replay_monitor.begin_window("window5", sync_cuda=False)
    replay_monitor.begin_window("window6", sync_cuda=False)
    measurement = replay_monitor.end_window("window6", sync_cuda=False)
    assert_measurement("window6", measurement, begin_time=26, elapsed_time=3, assert_calls=False)
    measurement = replay_monitor.end_window("window5", sync_cuda=False)
    assert_measurement("window5", measurement, begin_time=25, elapsed_time=8, assert_calls=False)
