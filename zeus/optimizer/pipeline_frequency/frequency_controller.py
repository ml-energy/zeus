"""Controller that sets the GPU's frequency in a non-blocking fashion."""

from __future__ import annotations

import atexit
import contextlib
import multiprocessing as mp

from zeus.device import get_gpus
from zeus.device.gpu import ZeusGPUNotSupportedError


class FrequencyController:
    """Spawns a separate process that sets the GPU frequency."""

    def __init__(self, device_id: int = 0) -> None:
        """Instantiate the frequency controller.

        Args:
            device_id: Device ID of the GPU to control.
        """
        self._q: mp.Queue[int | None] = mp.Queue()
        self._proc = mp.Process(target=self._controller_process, args=(device_id,))

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
        gpus = get_gpus()
        # Return the power limit to the default.
        gpus.resetPowerManagementLimit(device_id)

        # Set the memory frequency to be the highest.
        max_mem_freq = max(gpus.getSupportedMemoryClocks(device_id))
        with contextlib.suppress(ZeusGPUNotSupportedError):
            gpus.setMemoryLockedClocks(device_id, max_mem_freq, max_mem_freq)

        # Set the SM frequency to be the highest.
        max_freq = max(gpus.getSupportedGraphicsClocks(device_id, max_mem_freq))
        gpus.setGpuLockedClocks(device_id, max_freq, max_freq)
        current_freq = max_freq

        # Wait on the queue for the next frequency to set.
        while True:
            target_freq = self._q.get(block=True)
            if target_freq is None:
                break
            if current_freq != target_freq:
                gpus.setGpuLockedClocks(device_id, target_freq, target_freq)
                current_freq = target_freq

        # Reset everything.
        with contextlib.suppress(ZeusGPUNotSupportedError):
            gpus.resetMemoryLockedClocks(device_id)
        gpus.resetGpuLockedClocks(device_id)
