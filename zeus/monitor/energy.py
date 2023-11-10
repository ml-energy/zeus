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

"""Measure the GPU time and energy consumption of a block of code."""

from __future__ import annotations

import os
import atexit
from time import time
from pathlib import Path
from dataclasses import dataclass
from functools import cached_property, lru_cache

import pynvml

from zeus.monitor.power import PowerMonitor
from zeus.util.logging import get_logger
from zeus.util.framework import cuda_sync
from zeus.util.env import resolve_gpu_indices

logger = get_logger(__name__)


@dataclass
class Measurement:
    """Measurement result of one window.

    Attributes:
        time: Time elapsed (in seconds) during the measurement window.
        energy: Maps GPU indices to the energy consumed (in Joules) during the
            measurement window. GPU indices are from the DL framework's perspective
            after applying `CUDA_VISIBLE_DEVICES`.
    """

    time: float
    energy: dict[int, float]

    @cached_property
    def total_energy(self) -> float:
        """Total energy consumed (in Joules) during the measurement window."""
        return sum(self.energy.values())


class ZeusMonitor:
    """Measure the GPU energy and time consumption of a block of code.

    Works for multi-GPU and heterogeneous GPU types. Aware of `CUDA_VISIBLE_DEVICES`.
    For instance, if `CUDA_VISIBLE_DEVICES=2,3`, GPU index `1` passed into `gpu_indices`
    will be interpreted as CUDA device `3`.

    You can mark the beginning and end of a measurement window, during which the GPU
    energy and time consumed will be recorded. Multiple concurrent measurement windows
    are supported.

    For Volta or newer GPUs, energy consumption is measured very cheaply with the
    `nvmlDeviceGetTotalEnergyConsumption` API. On older architectures, this API is
    not supported, so a separate Python process is used to poll `nvmlDeviceGetPowerUsage`
    to get power samples over time, which are integrated to compute energy consumption.

    ## Integration Example

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
    print(f"Training took {result.time} seconds.")
    print(f"Training consumed {result.total_energy} Joules.")
    for gpu_idx, gpu_energy in result.energy.items():
        print(f"GPU {gpu_idx} consumed {gpu_energy} Joules.")
    ```

    Attributes:
        gpu_indices (`list[int]`): Indices of all the CUDA devices to monitor, from the
            DL framework's perspective after applying `CUDA_VISIBLE_DEVICES`.
        nvml_gpu_indices (`list[int]`): Indices of all the CUDA devices to monitor, from
            NVML/system's perspective.
    """

    def __init__(
        self,
        gpu_indices: list[int] | None = None,
        approx_instant_energy: bool = False,
        log_file: str | Path | None = None,
    ) -> None:
        """Instantiate the monitor.

        Args:
            gpu_indices: Indices of all the CUDA devices to monitor. Time/Energy measurements
                will begin and end at the same time for these GPUs (i.e., synchronized).
                If None, all the GPUs available will be used. `CUDA_VISIBLE_DEVICES`
                is respected if set, e.g., GPU index `1` passed into `gpu_indices` when
                `CUDA_VISIBLE_DEVICES=2,3` will be interpreted as CUDA device `3`.
                `CUDA_VISIBLE_DEVICES`s formatted with comma-separated indices are supported.
            approx_instant_energy: When the execution time of a measurement window is
                shorter than the NVML energy counter's update period, energy consumption may
                be observed as zero. In this case, if `approx_instant_energy` is True, the
                window's energy consumption will be approximated by multiplying the current
                instantaneous power consumption with the window's execution time. This should
                be a better estimate than zero, but it's still an approximation.
            log_file: Path to the log CSV file. If `None`, logging will be disabled.
        """
        # Save arguments.
        self.approx_instant_energy = approx_instant_energy

        # Initialize NVML.
        pynvml.nvmlInit()
        atexit.register(pynvml.nvmlShutdown)

        # CUDA GPU indices and NVML GPU indices are different if `CUDA_VISIBLE_DEVICES` is set.
        self.gpu_indices, self.nvml_gpu_indices = resolve_gpu_indices(gpu_indices)

        # Save all the NVML GPU handles. These should be called with system-level GPU indices.
        self.gpu_handles: dict[int, pynvml.c_nvmlDevice_t] = {}
        for nvml_gpu_index, gpu_index in zip(self.nvml_gpu_indices, self.gpu_indices):
            handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_gpu_index)
            self.gpu_handles[gpu_index] = handle

        # Initialize loggers.
        if log_file is None:
            self.log_file = None
        else:
            if dir := os.path.dirname(log_file):
                os.makedirs(dir, exist_ok=True)
            self.log_file = open(log_file, "w")
            logger.info("Writing measurement logs to %s.", log_file)
            self.log_file.write(
                f"start_time,window_name,elapsed_time,{','.join(map(lambda i: f'gpu{i}_energy', self.gpu_indices))}\n",
            )
            self.log_file.flush()

        logger.info("Monitoring GPU indices %s.", self.gpu_indices)

        # A dictionary that maps the string keys of active measurement windows to
        # the state of the measurement window. Each element in the dictionary is a tuple of:
        #     1) Time elapsed at the beginning of this window.
        #     2) Total energy consumed by each >= Volta GPU at the beginning of
        #        this window (`None` for older GPUs).
        self.measurement_states: dict[str, tuple[float, dict[int, float]]] = {}

        # Initialize power monitors for older architecture GPUs.
        old_gpu_indices = [
            gpu_index
            for gpu_index, is_new in zip(self.gpu_indices, self._is_new_arch_flags)
            if not is_new
        ]
        if old_gpu_indices:
            self.power_monitor = PowerMonitor(
                gpu_indices=old_gpu_indices, update_period=None
            )
        else:
            self.power_monitor = None

    @lru_cache
    def _is_new_arch(self, gpu: int) -> bool:
        """Return whether the GPU is Volta or newer."""
        return (
            pynvml.nvmlDeviceGetArchitecture(self.gpu_handles[gpu])
            >= pynvml.NVML_DEVICE_ARCH_VOLTA
        )

    @cached_property
    def _is_new_arch_flags(self) -> list[bool]:
        """A list of flags indicating whether each GPU is Volta or newer."""
        return [self._is_new_arch(gpu) for gpu in self.gpu_handles]

    def _get_instant_power(self) -> tuple[dict[int, float], float]:
        """Measure the power consumption of all GPUs at the current time."""
        power_measurement_start_time: float = time()
        power = {
            i: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            for i, h in self.gpu_handles.items()
        }
        power_measurement_time = time() - power_measurement_start_time
        return power, power_measurement_time

    def begin_window(self, key: str, sync_cuda: bool = True) -> None:
        """Begin a new measurement window.

        Args:
            key: Unique name of the measurement window.
            sync_cuda: Whether to synchronize CUDA before starting the measurement window.
                (Default: `True`)
        """
        # Make sure the key is unique.
        if key in self.measurement_states:
            raise ValueError(f"Measurement window '{key}' already exists")

        # Call cudaSynchronize to make sure we freeze at the right time.
        if sync_cuda:
            for gpu_index in self.gpu_handles:
                cuda_sync(gpu_index)

        # Freeze the start time of the profiling window.
        timestamp: float = time()
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
        logger.debug("Measurement window '%s' started.", key)

    def end_window(
        self, key: str, sync_cuda: bool = True, cancel: bool = False
    ) -> Measurement:
        """End a measurement window and return the time and energy consumption.

        Args:
            key: Name of an active measurement window.
            sync_cuda: Whether to synchronize CUDA before ending the measurement window.
                (default: `True`)
            cancel: Whether to cancel the measurement window. If `True`, the measurement
                window is assumed to be cancelled and discarded. Thus, an empty Measurement
                object will be returned and the measurement window will not be recorded in
                the log file either. `sync_cuda` is still respected.
        """
        # Retrieve the start time and energy consumption of this window.
        try:
            start_time, start_energy = self.measurement_states.pop(key)
        except KeyError:
            raise ValueError(f"Measurement window '{key}' does not exist") from None

        # Take instant power consumption measurements.
        # This, in theory, is introducing extra NVMLs call in the critical path
        # even if computation time is not so short. However, it is reasonable to
        # expect that computation time would be short if the user explicitly
        # turned on the `approx_instant_energy` option. Calling this function
        # as early as possible will lead to more accurate energy approximation.
        power, power_measurement_time = (
            self._get_instant_power() if self.approx_instant_energy else ({}, 0.0)
        )

        # Call cudaSynchronize to make sure we freeze at the right time.
        if sync_cuda:
            for gpu_index in self.gpu_handles:
                cuda_sync(gpu_index)

        # If the measurement window is cancelled, return an empty Measurement object.
        if cancel:
            logger.debug("Measurement window '%s' cancelled.", key)
            return Measurement(time=0.0, energy={gpu: 0.0 for gpu in self.gpu_handles})

        end_time: float = time()
        time_consumption: float = end_time - start_time
        energy_consumption: dict[int, float] = {}
        for gpu_index, gpu_handle in self.gpu_handles.items():
            # Query energy directly if the GPU has newer architecture.
            if self._is_new_arch(gpu_index):
                end_energy = (
                    pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle) / 1000.0
                )
                energy_consumption[gpu_index] = end_energy - start_energy[gpu_index]

        # If there are older GPU architectures, the PowerMonitor will take care of those.
        if self.power_monitor is not None:
            energy = self.power_monitor.get_energy(start_time, end_time)
            # Fallback to the instant power measurement if the PowerMonitor does not
            # have the power samples.
            if energy is None:
                energy = {gpu: 0.0 for gpu in self.power_monitor.gpu_indices}
            energy_consumption |= energy

        # Approximate energy consumption if the measurement window is too short.
        if self.approx_instant_energy:
            for gpu_index in self.gpu_indices:
                if energy_consumption[gpu_index] == 0.0:
                    energy_consumption[gpu_index] = power[gpu_index] * (
                        time_consumption - power_measurement_time
                    )

        logger.debug("Measurement window '%s' ended.", key)

        # Add to log file.
        if self.log_file is not None:
            self.log_file.write(
                f"{start_time},{key},{time_consumption},"
                + ",".join(str(energy_consumption[gpu]) for gpu in self.gpu_indices)
                + "\n"
            )
            self.log_file.flush()

        return Measurement(time_consumption, energy_consumption)
