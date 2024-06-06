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

"""Utilities for testing."""

from __future__ import annotations

from pathlib import Path

from zeus.monitor import Measurement, ZeusMonitor
from zeus.utils.framework import cuda_sync
from zeus.utils.logging import get_logger


class ReplayZeusMonitor(ZeusMonitor):
    """A mock ZeusMonitor that replays windows recorded by a real monitor.

    This class is for testing only. Based on a CSV log file that records the time
    and energy measurements of `ZeusMonitor` measurement windows, users can drop-in
    replace `ZeusMonitor` with this class to replay the measurement windows and
    *fast forward* training and time/energy measurement.

    The methods exposed is identical to or a superset of `ZeusMonitor`, but behaves
    differently. Instead of monitoring the GPU, it replays events from a log file.
    The log file generated by `ZeusMonitor` (`log_file`) is guaranteed to be compatible
    and will replay time and energy measurements just like how the real monitor
    experienced them. Note that in the case of concurrent ongoing measurement windows,
    the log rows file should record windows in the order of `end_window` calls.

    Attributes:
        gpu_indices (`list[int]`): Indices of all the CUDA devices to monitor.
    """

    def __init__(
        self,
        gpu_indices: list[int] | None = None,
        approx_instant_energy: bool = False,
        log_file: str | Path | None = None,
        ignore_sync_cuda: bool = False,
        match_window_name: bool = True,
    ) -> None:
        """Initialize the replay monitor.

        The log file should be a CSV file with the following header (e.g. gpu_indices=[0, 2]):
        `start_time,window_name,elapsed_time,gpu0_energy,gpu2_energy`

        Args:
            gpu_indices: Indices of all the CUDA devices to monitor. This should be consistent
                with the indices used in the log file. If `None`, GPU indices will be inferred
                from the log file header. Does not respect `CUDA_VISIBLE_DEVICES`.
                (Default: `None`)
            approx_instant_energy: Whether to approximate the instant energy consumption. Not used.
            log_file: Path to the log CSV file to replay events from. `None` is not allowed.
            ignore_sync_cuda: Whether to ignore `sync_cuda` calls. (Default: `False`)
            match_window_name: Whether to make sure window names match. (Default: `True`)
        """
        if log_file is None:
            raise ValueError("`log_file` cannot be `None` for `ReplayZeusMonitor`.")

        self.approx_instant_energy = approx_instant_energy
        self.log_file = open(log_file)
        self.ignore_sync_cuda = ignore_sync_cuda
        self.match_window_name = match_window_name

        # Infer GPU indices from the log file if not provided.
        header = self.log_file.readline()
        if gpu_indices is None:
            gpu_indices = [
                int(gpu.split("_")[0][3:]) for gpu in header.split(",")[3:] if gpu
            ]
        self.nvml_gpu_indices = self.gpu_indices = gpu_indices

        self.logger = get_logger(type(self).__name__)
        self.logger.info(
            "Replaying from '%s' with GPU indices %s", log_file, gpu_indices
        )

        # Keep track of ongoing measurement windows.
        self.ongoing_windows = []

    def begin_window(self, key: str, sync_cuda: bool = True) -> None:
        """Begin a new window.

        This method just pushes the key into a list of ongoing measurement windows,
        and just makes sure it's unique.

        Args:
            key: Name of the measurement window.
            sync_cuda: Whether to synchronize CUDA before starting the measurement window.
                (Default: `True`)
        """
        if key in self.ongoing_windows:
            raise RuntimeError(f"Window {key} is already ongoing.")
        self.ongoing_windows.append(key)

        if not self.ignore_sync_cuda and sync_cuda:
            for gpu_index in self.gpu_indices:
                cuda_sync(gpu_index)

        self.logger.info("Measurement window '%s' started.", key)

    def end_window(
        self, key: str, sync_cuda: bool = True, cancel: bool = False
    ) -> Measurement:
        """End an ongoing window.

        This method pops the key from a list of ongoing measurement windows and
        constructs a `Measurement` object corresponding to the name of the window
        from the log file. If the name of the window does not match the expected
        one, a `RuntimeError` is raised.

        Args:
            key: Name of the measurement window.
            sync_cuda: Whether to synchronize CUDA before ending the measurement window.
                (Default: `True`)
            cancel: Whether to cancel the measurement window. This will not consume a
                line from the log file. (Default: `False`)
        """
        try:
            self.ongoing_windows.remove(key)
        except ValueError:
            raise RuntimeError(f"Window {key} is not ongoing.") from None

        if not self.ignore_sync_cuda and sync_cuda:
            for gpu_index in self.gpu_indices:
                cuda_sync(gpu_index)

        if cancel:
            self.logger.info("Measurement window '%s' cancelled.", key)
            return Measurement(
                time=0.0,
                gpu_energy={gpu_index: 0.0 for gpu_index in self.gpu_indices},
                cpu_energy={cpu_index: 0.0 for cpu_index in self.cpu_indices},
            )

        # Read the next line from the log file.
        line = self.log_file.readline()
        if not line:
            raise RuntimeError("No more lines in the log file.")
        _, window_name, *nums = line.split(",")
        if self.match_window_name and window_name != key:
            raise RuntimeError(f"Was expecting {window_name}, not {key}.")
        if len(nums) != len(self.gpu_indices) + 1:
            raise RuntimeError(
                f"Line has unexpected number of energy measurements: {line}"
            )
        time_consumption, *energy_consumptions = map(float, nums)
        energy = dict(zip(self.gpu_indices, energy_consumptions))
        measurement = Measurement(
            time=time_consumption, gpu_energy=energy, cpu_energy={}
        )

        self.logger.info("Measurement window '%s' ended (%s).", key, measurement)

        return measurement
