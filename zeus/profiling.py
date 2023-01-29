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

import time
import subprocess
import signal
import atexit
import logging
from typing import List, Union, Tuple

import pynvml

from zeus import analyze
from zeus.util.logging import LOG  # TODO: Refactor to per-module logger.


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
    time_consumed, energy_consumed = prof_service.pop_window()

    print(f"Training takes {time_consumed} seconds.")
    for gpu_idx, energy_consumed_per_gpu in energy_consumed:
        print(f"GPU {gpu_idx} consumes {energy_consumed_per_gpu} Joules.")
    ```
    Please checkout out [zeus.run.dataloader] for a complete integrated example.
    """

    def __init__(
        self,
        gpu_handles: List[pynvml.c_nvmlDevice_t],
        monitor_path: str | None = None,
        power_log_prefix: str | None = None,
    ) -> None:
        """Instantiate the profiling service. Check the chip architecture and decide our profiling method.

        Args:
            gpu_handles: Handles of all the devices.
            monitor_path: The path to zeus monitor executable.
            power_log_prefix: The prefix of power logging file.

        Raises:
            ValueError: Profiling on GPUs with architecture older than Nvidia Volta requires
                zeus monitor for power profiling. Raises when either `monitor_path` or
                `power_log_prefix` is not provided.
        """
        self.gpu_handles: List[pynvml.c_nvmlDevice_t] = gpu_handles
        self.is_newer_arch: List[bool] = [False for _ in range(len(gpu_handles))]

        # Check the chip architecture of all the gpus
        for gpu_idx, handle in enumerate(self.gpu_handles):
            arch = pynvml.nvmlDeviceGetArchitecture(handle)
            self._log(f"Architecture: {NVML_DEVICE_ARCH_MAPPING[arch]}", gpu_idx)
            if arch >= pynvml.NVML_DEVICE_ARCH_VOLTA:
                self.is_newer_arch[gpu_idx] = True
            else:
                if monitor_path is None or power_log_prefix is None:
                    raise ValueError(
                        "`monitor_path` and `power_log_prefix` must be provided if "
                        "you are using chip architecture before Nvidia Volta."
                    )

        self.monitor_path = monitor_path
        self.power_log_prefix = power_log_prefix

        # A stack that maintains the information at the start point of uncompleted profiling windows.
        # Each element in the stack is a tuple `(prof_start_time, prof_start_energy)` where
        # `prof_start_energy` is a list that stores energy consumed when the window starts at each GPUs
        # with newer architecture.
        self.prof_start_info: List[Tuple[float, List[float]]] = []

        # Start monitors to polling power for the GPUs with older architecture
        self.monitors: List[Union[subprocess.Popen, None]] = [
            None for _ in range(len(gpu_handles))
        ]
        self._start_monitors()

        # Kill the monitors when the training script exits
        def exit_hook():
            self._stop_monitors()

        atexit.register(exit_hook)

    def _power_log_path(self, gpu_idx: int) -> None:
        """Return the path of power log file for one gpu.

        Args:
            gpu_idx: The index of GPU.
        """
        return f"{self.power_log_prefix}+gpu{gpu_idx}.power.csv"

    def _start_monitors(self) -> None:
        """Spawn monitor processes for power polling for GPUs with older architecture."""
        for gpu_idx in range(len(self.gpu_handles)):
            if not self.is_newer_arch[gpu_idx]:
                monitor = subprocess.Popen(
                    [
                        self.monitor_path,
                        self._power_log_path(gpu_idx),
                        "0",
                        "100",
                        str(gpu_idx),
                    ],
                )
                self.monitors[gpu_idx] = monitor
                self._log(f"[GPU_{gpu_idx}] Zeus monitor started.", gpu_idx)

    def _stop_monitors(self) -> None:
        """Kill the power monitor subprocess."""
        for gpu_idx in range(len(self.gpu_handles)):
            if not self.is_newer_arch[gpu_idx]:
                # Sanity check that monitor exists for GPU with older architecture
                assert (
                    self.monitors[gpu_idx] is not None
                ), f"monitor is not spawned for GPU_{gpu_idx}"
                self.monitors[gpu_idx].send_signal(signal.SIGINT)
        for gpu_idx in range(len(self.gpu_handles)):
            if not self.is_newer_arch[gpu_idx]:
                self.monitors[gpu_idx].wait(timeout=1.0)
                self.monitors[gpu_idx].kill()
                self._log(f"[GPU_{gpu_idx}] Zeus monitor stopped.", gpu_idx)

    def push_window(self) -> None:
        """Push one profiling window to the stack."""
        prof_start_energy: List[float] = [0 for _ in range(len(self.gpu_handles))]
        prof_start_time: float = time.monotonic()
        for gpu_idx, handle in enumerate(self.gpu_handles):
            if self.is_newer_arch[gpu_idx]:
                # Query NVML energy method for the energy consumed until
                # the start of profiling window.
                prof_start_energy[gpu_idx] = self._millijoules_to_joules(
                    pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                )

        # Push the profiling start information to the stack
        self.prof_start_info.append((prof_start_time, prof_start_energy))

    def pop_window(self) -> Tuple[float, List[float]]:
        """Pop one profiling window out of the stack, returns the time consumption and energy consumption for each GPU.

        Returns:
            A tuple `(time_consumed, list_of_energy_consumed_per_gpu)`.
        """
        if not self.prof_start_info:
            raise RuntimeError(
                "No profiling window exists. Consider calling `push_window` first."
            )
        prof_start_time, prof_start_energy = self.prof_start_info.pop()
        prof_end_time = time.monotonic()
        time_consumed = prof_end_time - prof_start_time
        energy_consumed: List[float] = [0 for _ in range(len(self.gpu_handles))]
        for gpu_idx, handle in enumerate(self.gpu_handles):
            if self.is_newer_arch[gpu_idx]:
                prof_end_energy = self._millijoules_to_joules(
                    pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                )
                energy_consumed[gpu_idx] = prof_end_energy - prof_start_energy[gpu_idx]
            else:
                energy_consumed[gpu_idx] = analyze.energy(
                    self._power_log_path(gpu_idx), prof_start_time, prof_end_time
                )

        prof_result = (
            time_consumed,
            [energy_consumed[gpu_idx] for gpu_idx in range(len(self.gpu_handles))],
        )
        return prof_result

    def _millijoules_to_joules(self, millijoules: int) -> float:
        """Convert millijoules to joules.

        Args:
            millijoules: Energy in millijoules

        Returns:
            Energy in joules.
        """
        return millijoules / 1000.0

    def _log(
        self, message: str, gpu_idx: int | None = None, level: int = logging.INFO
    ) -> None:
        """Print out message with prefix.

        Args:
            message: The message to log out.
            gpu_idx: The index of GPU for GPU-level logging. Should be `None`
                when logging global information. (Default: `None`)
            level: The logging level to use. (Default: `logging.INFO`)
        """
        log_prefix = "[ZeusProfilingService]"
        if gpu_idx is not None:
            # GPU-level logging
            gpu_log_prefix = f"[GPU_{gpu_idx}]"
            LOG.log(level, "%s %s %s", log_prefix, gpu_log_prefix, message)
        else:
            # Global logging
            LOG.log(level, "%s %s", log_prefix, message)


NVML_DEVICE_ARCH_MAPPING = {
    pynvml.NVML_DEVICE_ARCH_KEPLER: "KEPLER",
    pynvml.NVML_DEVICE_ARCH_MAXWELL: "MAXWELL",
    pynvml.NVML_DEVICE_ARCH_PASCAL: "PASCAL",
    pynvml.NVML_DEVICE_ARCH_VOLTA: "VOLTA",
    pynvml.NVML_DEVICE_ARCH_TURING: "TURING",
    pynvml.NVML_DEVICE_ARCH_AMPERE: "AMPERE",
    pynvml.NVML_DEVICE_ARCH_UNKNOWN: "UNKNOWN",
}
