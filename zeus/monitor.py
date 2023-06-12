# Copyright (C) 2022 Jae-Won Chung <jwnchung@umich.edu>
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

"""Measure the GPU time and energy consumption of a block of code."""

from __future__ import annotations

import os
import time
import shutil
import signal
import atexit
import logging
import tempfile
import subprocess
from functools import cached_property, lru_cache
from dataclasses import dataclass

import pynvml

from zeus import analyze
from zeus.util.framework import cuda_sync
from zeus.util.logging import get_logger


@dataclass
class Measurement:
    """Measurement result of one window.

    Attributes:
        time: Time elapsed (in seconds) during the measurement window.
        energy: Maps GPU indices to the energy consumed (in Joules) during the
            measurement window.
    """

    time: float
    energy: dict[int, float]

    @cached_property
    def total_energy(self) -> float:
        """Total energy consumed (in Joules) during the measurement window."""
        return sum(self.energy.values())


class ZeusMonitor:
    """Measure the GPU energy and time consumption of a block of code.

    Works for multi-GPU, heterogeneous GPU types, and any DL framework.

    You can mark the beginning and end of a measurement window, during which the GPU
    energy and time consumed will be recorded. Multiple concurrent measurement windows
    are supported.

    ## Integrated Example

    ```python
    from zeus.monitor import ZeusMontior

    # Time/Energy measurements for four GPUs will begin and end at the same time.
    gpu_indices = [0, 1, 2, 3]
    monitor = ZeusMonitor(gpu_indices)

    # Mark the beginning of a measurement window. You can use any string
    # as the window name, but make sure it's unique.
    monitor.begin_window("entire_training")

    # Actual work
    training(x, y)

    # Mark the end of a measurement window and retrieve the measurment result.
    result = monitor.end_window("entire_training")

    # Print the measurement result.
    time_consumed, energy_consumed = prof_result.time, prof_result.energy
    print(f"Training took {time_consumed} seconds.")
    for gpu_idx, gpu_energy in zip(gpu_indices, energy_consumed):
        print(f"GPU {gpu_idx} consumed {gpu_energy} Joules.")
    ```
    """

    def __init__(
        self,
        gpu_indices: list[int] | None = None,
        monitor_exec: str = "zeus_monitor",
    ) -> None:
        """Instantiate the monitor.

        For Volta or newer GPUs, energy consumption is measured very cheaply with the
        `nvmlDeviceGetTotalEnergyConsumption` API. Otherwise, the energy API is not
        supported. Thus, the Zeus monitor binary is used to poll `nvmlDeviceGetPowerUsage`
        and write to a temporary CSV file, which is then integrated over time to compute
        energy consumption.

        Args:
            gpu_indices: Indices of all the CUDA devices to monitor. Time/Energy measurement
                will begin and end at the same time for these GPUs (i.e., synchronized).
                If None, all the GPUs available will be used (while respecting the
                `CUDA_VISIBLE_DEVICES` environment variable). (Default: `None`)
            monitor_exec: Zeus monitor executable. (Default: `"zeus_monitor"`)
        """
        # Initialize NVML.
        pynvml.nvmlInit()

        # Initialize logger.
        self.logger = get_logger(type(self).__name__)

        # If `gpu_indices` is None, use all the GPUs available.
        if gpu_indices is None:
            # NVML is not aware of the `CUDA_VISIBLE_DEVICES` environment variable,
            # so we need to manually check whether it's set.
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices is not None:
                gpu_indices = [int(idx) for idx in cuda_visible_devices.split(",")]
            else:
                gpu_indices = list(range(pynvml.nvmlDeviceGetCount()))

        # Save all the GPU handles.
        self.gpu_handles: dict[int, pynvml.c_nvmlDevice_t] = {}
        for gpu_index in gpu_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.gpu_handles[gpu_index] = handle

        # A dictionary that maps the string keys of active measurement windows to
        # the state of the measurement window. Each element in the dictionary is a tuple of:
        #     1) Time elapsed at the beginning of this window.
        #     2) Total energy consumed by each >= Volta GPU at the beginning of this window.
        self.measurement_states: dict[str, tuple[float, dict[int, float]]] = {}

        # Shutdown NVML and kill the monitors when the training script exits.
        # The exit hook is a no-op when no monitors are running, so we can register it
        # before starting any monitors.
        def exit_hook():
            # Shutdown NVML.
            pynvml.nvmlShutdown()
            self._stop_monitors()

        atexit.register(exit_hook)

        # Start monitors that poll power for older architecture GPUs.
        self.monitors: dict[int, subprocess.Popen] = {}
        self._start_monitors(monitor_exec)

    @lru_cache
    def _monitor_log_path(self, gpu_index: int) -> str:
        """Get the path of the monitor log file for the given GPU index."""
        return os.path.join(self.monitor_log_dir, f"gpu{gpu_index}.power.csv")

    def _start_monitors(self, monitor_exec: str) -> None:
        """Spawn monitor processes for power polling for GPUs with older architecture.

        Raises:
            `ValueError`: If `monitor_path` is not executable when there exists GPUs with
                older architecture and monitors should be spawned for power polling.
        """
        old_arch_flags = [
            not self._is_new_arch(gpu_index) for gpu_index in self.gpu_handles
        ]
        # At least one GPU has an old architecture and we need to spawn the Zeus monitor
        # for that GPU.
        if any(old_arch_flags):
            # Check whether the monitor path is good.
            if not shutil.which(monitor_exec):
                raise ValueError(f"'{monitor_exec}' is not executable")
            # Create a temporary directory.
            self.monitor_log_dir = tempfile.mkdtemp()

        # Capture the time when we started the monitors.
        self.monitor_start_time = time.monotonic()

        # Spawn monitor process when GPU has older architecture.
        for gpu_index, arch_is_old in zip(self.gpu_handles, old_arch_flags):
            if arch_is_old:
                log_path = self._monitor_log_path(gpu_index)
                # 10 Hz (100 ms sleep) polling should be enough.
                monitor = subprocess.Popen(
                    [monitor_exec, log_path, "0", "100", str(gpu_index)],
                )
                # Save the mapping from `gpu_index` to monitor.
                self.monitors[gpu_index] = monitor
                self._log("Zeus monitor started.", gpu_index)

    def _stop_monitors(self) -> None:
        """Kill the power monitor subprocess."""
        for monitor in self.monitors.values():
            monitor.send_signal(signal.SIGINT)
        for gpu_index, monitor in self.monitors.items():
            monitor.wait(timeout=1.0)
            monitor.kill()
            self._log("Zeus monitor stopped.", gpu_index)

    @lru_cache
    def _is_new_arch(self, gpu: int) -> bool:
        """Check whether the given GPU is Volta or newer."""
        gpu_handle = self.gpu_handles[gpu]
        return (
            pynvml.nvmlDeviceGetArchitecture(gpu_handle)
            >= pynvml.NVML_DEVICE_ARCH_VOLTA
        )

    def begin_window(self, key: str, sync_cuda: bool = True) -> None:
        """Begin a new measurement window.

        Args:
            key: Unique name of the measurement window.
            sync_cuda: Whether to synchronize CUDA before starting the measurement window.
        """
        # Make sure the key is unique.
        if key in self.measurement_states:
            raise ValueError(f"Measurement window '{key}' already exists")

        # Call cudaSynchronize to make sure we freeze at the right time.
        if sync_cuda:
            for gpu_index in self.gpu_handles:
                cuda_sync(gpu_index)

        # Freeze the start time of the profiling window.
        timestamp: float = time.monotonic()
        energy_state: dict[int, float] = {}
        for gpu_index, gpu_handle in self.gpu_handles.items():
            # Query energy directly if the GPU has newer architecture.
            # Otherwise, the Zeus power monitor is running in the background to
            # collect power consumption, so we just need to read the log file later.
            if self._is_new_arch(gpu_index):
                energy_state[gpu_index] = (
                    pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle) / 1000.0
                )

        # Add measurement state to dictionary.
        self.measurement_states[key] = (timestamp, energy_state)
        self._log(f"Measurement window '{key}' started.")

    def end_window(self, key: str, sync_cuda: bool = True) -> Measurement:
        """End a measurement window and return the time and energy consumption.

        Args:
            key: Name of an active measurement window.
            sync_cuda: Whether to synchronize CUDA before ending the measurement window.
                (default: `True`)
        """
        # Retrieve the start time and energy consumption of this window.
        try:
            start_time, start_energy = self.measurement_states.pop(key)
        except KeyError:
            raise ValueError(f"Measurement window '{key}' does not exist") from None

        # Call cudaSynchronize to make sure we freeze at the right time.
        if sync_cuda:
            for gpu_index in self.gpu_handles:
                cuda_sync(gpu_index)

        end_time: float = time.monotonic()
        time_consumption: float = end_time - start_time
        energy_consumption: dict[int, float] = {}
        for gpu_index, gpu_handle in self.gpu_handles.items():
            # Query energy directly if the GPU has newer architecture.
            if self._is_new_arch(gpu_index):
                end_energy = (
                    pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle) / 1000.0
                )
                energy_consumption[gpu_index] = end_energy - start_energy[gpu_index]
            # Otherwise, read the log file to compute energy consumption.
            else:
                energy_consumption[gpu_index] = analyze.energy(
                    self._monitor_log_path(gpu_index),
                    start_time - self.monitor_start_time,
                    end_time - self.monitor_start_time,
                )

        self._log(f"Measurement window '{key}' ended.")
        return Measurement(time_consumption, energy_consumption)

    def _log(
        self, message: str, gpu_index: int | None = None, level: int = logging.INFO
    ) -> None:
        """Print out message with prefix.

        Args:
            message: The message to log out.
            gpu_index: The index of GPU for GPU-level logging. Should be `None`
                when logging global information. (Default: `None`)
            level: The logging level to use. (Default: `logging.INFO`)
        """
        if gpu_index is not None:
            message = f"[GPU {gpu_index}] {message}"
        self.logger.log(level, message)
