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

"""Helpers for profiling energy and time inside a training script."""

from __future__ import annotations

import os
import time
import subprocess
import signal
import atexit
import logging
import tempfile
from dataclasses import dataclass

import pynvml
import torch

from zeus import analyze

LOG = logging.getLogger(__name__)


@dataclass
class ProfilingResult:
    """Profiling result of one window.

    Args:
        time: Time elapsed (in seconds) within the profiling window.
        energy: A list of energy consumption (in Joules) within the profiling window for each GPU.
    """

    time: float
    energy: list[float]


class ZeusProfilingService:
    """Profiles and energy and time inside a training script.

    Push a profiling window to start profiling and pop to get the profiling result
    of the *last* pushed profiling window. Multiple nested profiling windows are
    supported using a stack.

    ## Integrated Example

    ```python
    prof_service = zeus.profiling.ZeusProfilingService(gpu_handles, monitor_path, power_log_prefix)

    # Push a profile window
    prof_service.push_window()

    # Actual work
    training(x, y)

    # Pop a profile window and get the profiling result
    prof_result = prof_service.pop_window()
    time_consumed, energy_consumed = prof_result.time, prof_result.energy

    print(f"Training takes {time_consumed} seconds.")
    for gpu_idx, energy_consumed_per_gpu in energy_consumed:
        print(f"GPU {gpu_idx} consumes {energy_consumed_per_gpu} Joules.")
    ```

    Please checkout out [`ZeusDataLoader`][zeus.run.ZeusDataLoader] for a
    complete integrated example.
    """

    def __init__(
        self,
        gpu_indices: list[int] | None = None,
        monitor_path: str = "zeus_monitor",
        monitor_log_dir: str = "",
        monitor_log_prefix: str = "",
    ) -> None:
        """Instantiate the profiling service. Check the chip architecture and decide our profiling method.

        Args:
            gpu_indices: Indices of all the devices that will be used for training. If None, all the GPUs
                available will be used. (Default: `None`)
            monitor_path: The path to zeus monitor executable. (Default: `"zeus_monitor"`)
            monitor_log_dir: The directory where monitor logging files will be saved. If not provided, a
                temporary directory will be created. (Default: `""`)
            monitor_log_prefix: The prefix of monitor logging files for power polling. (Default: `""`)
        """
        # Initialize NVML.
        pynvml.nvmlInit()

        # Save attributes.
        self.monitor_path = monitor_path
        self.monitor_log_dir = monitor_log_dir
        self.monitor_log_prefix = monitor_log_prefix

        # If `gpu_indices` is None, use all the GPUs available.
        if gpu_indices is None:
            gpu_indices = list(range(pynvml.nvmlDeviceGetCount()))
        self.gpu_indices = gpu_indices

        # Save all the GPU handles.
        self.gpu_handles: list[pynvml.c_nvmlDevice_t] = []
        for gpu_index in self.gpu_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            # Set persistence mode.
            pynvml.nvmlDeviceSetPersistenceMode(handle, pynvml.NVML_FEATURE_ENABLED)
            self.gpu_handles.append(handle)

        # A stack that maintains the information at the start point of uncompleted profiling windows.
        # Each element in the stack is a tuple of:
        #     1) Time elapsed at the start of this profiling window.
        #     2) Energy consumed at each GPUs with newer architecture at the start of this profiling window.
        self.prof_start_info: list[tuple[float, dict[int, float]]] = []

        # Start monitors to polling power for the GPUs with older architecture.
        self.monitors: dict[int, subprocess.Popen] = {}
        self._start_monitors()

        # Shutdown NVML and kill the monitors when the training script exits.
        def exit_hook():
            # Shutdown NVML.
            pynvml.nvmlShutdown()
            self._stop_monitors()

        atexit.register(exit_hook)

    def _monitor_log_path(self, gpu_index: int) -> str:
        """Return the path of power log file for one gpu.

        Args:
            gpu_index: The index of GPU.
        """
        return (
            f"{self.monitor_log_dir}/{self.monitor_log_prefix}+gpu{gpu_index}.power.csv"
        )

    def _start_monitors(self) -> None:
        """Spawn monitor processes for power polling for GPUs with older architecture.

        Raises:
            `RuntimeError`: `self.monitor_path` is not executable when there exists GPUs with
                older architecture and monitors should be spawned for power polling.
        """
        arch_are_old = [
            pynvml.nvmlDeviceGetArchitecture(handle) < pynvml.NVML_DEVICE_ARCH_VOLTA
            for handle in self.gpu_handles
        ]
        # At least one GPU has an old architecture. Do the necessary setups.
        if any(arch_are_old):
            # Check whether the monitor path is good.
            if not os.access(self.monitor_path, os.X_OK):
                raise ValueError(f"'{self.monitor_path}' is not executable")
            if self.monitor_log_dir:
                # Create `monitor_log_dir` if it does not exist.
                os.makedirs(self.monitor_log_dir, exist_ok=True)
            else:
                # Create a temporary directory.
                self.monitor_log_dir = tempfile.mkdtemp()

        # Spawn monitor process when GPU has older architecture.
        for gpu_index, arch_is_old in zip(self.gpu_indices, arch_are_old):
            if arch_is_old:
                monitor = subprocess.Popen(
                    [
                        self.monitor_path,
                        self._monitor_log_path(gpu_index),
                        "0",
                        "100",
                        str(gpu_index),
                    ],
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

    def push_window(self) -> None:
        """Push one profiling window to the stack."""
        # Call cudaSynchronize to make sure we freeze at the right time.
        for gpu_index in self.gpu_indices:
            torch.cuda.synchronize(gpu_index)

        # Get the information at the start of profiling window.
        prof_start_energy: dict[int, float] = {}
        # Freeze the start time of the profiling window.
        prof_start_time: float = time.monotonic()
        for gpu_index, gpu_handle in zip(self.gpu_indices, self.gpu_handles):
            # If no monitor exists for this GPU, we need to save its energy consumed at the start of profiling window.
            if gpu_index not in self.monitors:
                # Query NVML energy method for the energy consumed until the start of profiling window.
                prof_start_energy[gpu_index] = (
                    pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle) / 1000.0
                )

        # Push the profiling start information to the stack.
        self.prof_start_info.append((prof_start_time, prof_start_energy))
        self._log("Profiling window pushed.")

    def pop_window(self) -> ProfilingResult:
        """Pop one profiling window out of the stack, returns the time consumption and energy consumption for each GPU.

        Returns:
            An object of `ProflilingResult` that contains time and energy consumption data.

        Raises:
            `RuntimeError`: The stack that stores profiling windows is empty. Users should always call `push_window()`
                before `pop_window()` to make sure the profiling window is set up correctly.
        """
        # Call cudaSynchronize to make sure we freeze at the right time.
        for gpu_index in self.gpu_indices:
            torch.cuda.synchronize(gpu_index)

        # Get the time at the end of the profiling window.
        prof_end_time: float = time.monotonic()

        # Check whether there is a profiling window in the stack.
        if not self.prof_start_info:
            raise RuntimeError(
                "No profiling window active. Consider calling `push_window` first."
            )

        # Pop out the info at the start of the profiling window, compute the time and energy
        # consumption for this profiling window.
        prof_start_time, prof_start_energy = self.prof_start_info.pop()
        time_consumed = prof_end_time - prof_start_time
        energy_consumed: list[float] = []
        for gpu_index, gpu_handle in zip(self.gpu_indices, self.gpu_handles):
            if gpu_index in self.monitors:
                # For GPUs with older architectures, compute the energy consumption from the power polling data.
                energy_consumed.append(
                    analyze.energy(
                        self._monitor_log_path(gpu_index),
                        prof_start_time,
                        prof_end_time,
                    )
                )
            else:
                # For GPUs with newer architectures, compute the energy consumption as the difference
                # at the start and end of the profiling window.
                prof_end_energy = (
                    pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle) / 1000.0
                )
                energy_consumed.append(prof_end_energy - prof_start_energy[gpu_index])

        self._log("Profiling window popped.")
        return ProfilingResult(time_consumed, energy_consumed)

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
        log_prefix = "[ZeusProfilingService]"
        if gpu_index is not None:
            # GPU-level logging
            gpu_log_prefix = f"[GPU_{gpu_index}]"
            LOG.log(level, "%s %s %s", log_prefix, gpu_log_prefix, message)
        else:
            # Global logging
            global_log_prefix = (
                f"[GPU_{','.join([str(gpu_index) for gpu_index in self.gpu_indices])}]"
            )
            LOG.log(level, "%s %s %s", log_prefix, global_log_prefix, message)
