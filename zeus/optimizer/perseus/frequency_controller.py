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

"""Controller that sets the GPU's frequency in a non-blocking fashion."""

from __future__ import annotations

import atexit
import contextlib
import multiprocessing as mp

import pynvml


class FrequencyController:
    """Spawns a separate process that sets the GPU frequency."""

    def __init__(self, nvml_device_id: int = 0) -> None:
        """Instantiate the frequency controller.

        Args:
            nvml_device_id: The NVML device ID of the GPU to control.
        """
        self._q: mp.Queue[int | None] = mp.Queue()
        self._proc = mp.Process(target=self._controller_process, args=(nvml_device_id,))

        atexit.register(self.end)
        self._proc.start()

    def set_frequency(self, frequency: int) -> None:
        """Set the GPU's frequency asynchronously.

        If `frequency` is zero, returns without doing anything.
        """
        if frequency != 0:
            self._q.put(frequency, block=False)

    def end(self) -> None:
        """Stop the controller process."""
        self._q.put(None, block=False)

    def _controller_process(self, device_id: int) -> None:
        """Receive frequency values through a queue and apply it."""
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

        # Return the power limit to the default.
        pynvml.nvmlDeviceSetPowerManagementLimit(
            handle,
            pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle),
        )

        # Set the memory frequency to be the highest.
        max_mem_freq = max(pynvml.nvmlDeviceGetSupportedMemoryClocks(handle))
        with contextlib.suppress(pynvml.NVMLError_NotSupported):  # type: ignore
            pynvml.nvmlDeviceSetMemoryLockedClocks(handle, max_mem_freq, max_mem_freq)

        # Set the SM frequency to be the highest.
        max_freq = max(
            pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, max_mem_freq)
        )
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, max_freq, max_freq)
        current_freq = max_freq

        # Wait on the queue for the next frequency to set.
        while True:
            target_freq = self._q.get(block=True)
            if target_freq is None:
                break
            if current_freq != target_freq:
                pynvml.nvmlDeviceSetGpuLockedClocks(handle, target_freq, target_freq)
                current_freq = target_freq

        # Reset everything.
        with contextlib.suppress(pynvml.NVMLError_NotSupported):  # type: ignore
            pynvml.nvmlDeviceResetMemoryLockedClocks(handle)
        pynvml.nvmlDeviceResetGpuLockedClocks(handle)
        pynvml.nvmlShutdown()
