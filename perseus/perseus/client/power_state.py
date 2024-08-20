"""Machinery for setting the GPU's power state (power limit or frequency) in a non-blocking fashion."""

from __future__ import annotations

import time
from typing import Literal
import multiprocessing as mp

import pynvml  # type: ignore

POWER_CONTROLLER: PowerController | None = None


def init_global_power_controller(
    default_power_state: int | None,
    output_dir: str | None,
    device_id: int = 0,
    power_control_mode: Literal["power limit", "frequency"] = "frequency",
) -> None:
    """Instantiate the global PowerController."""
    global POWER_CONTROLLER
    POWER_CONTROLLER = PowerController(
        default_power_state=default_power_state,
        output_dir=output_dir,
        device_id=device_id,
        power_control_mode=power_control_mode,
    )


def get_power_controller() -> PowerController:
    """Get the global PowerController instance."""
    if POWER_CONTROLLER is None:
        raise RuntimeError("First call `init_global_power_controller`.")
    return POWER_CONTROLLER


class PowerController:
    """Spawns a separate process that sets the GPU power state."""

    def __init__(
        self,
        default_power_state: int | None,
        output_dir: str | None,
        device_id: int = 0,
        power_control_mode: Literal["power limit", "frequency"] = "frequency",
    ) -> None:
        """Instantiate the power controller."""
        self.default_power_state = default_power_state
        self.output_dir = output_dir
        self.device_id = device_id
        self.power_control_mode = power_control_mode

        # pylint: disable=unsubscriptable-object
        self._q: mp.Queue[int | None] = mp.Queue()  # fmt: skip
        self._proc = mp.Process(
            target=self._frequency_manager_process
            if power_control_mode == "frequency"
            else self._power_limit_manager_process,
        )
        self._proc.start()

    def set(self, power_state: int) -> None:
        """Set the power state asynchronously.

        If `power_state` is zero, just returns.
        """
        if power_state != 0:
            self._q.put(power_state, block=False)

    def end(self) -> None:
        """Stop the process and have it generate the power state history CSV file."""
        self._q.put(None, block=False)

    def _frequency_manager_process(self) -> None:
        """Receive frequency values through a queue and apply it.

        Records the start/end time of all frequencies and outputs events as a CSV file.
        """
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)

        # Reset any power limit settings.
        pynvml.nvmlDeviceSetPowerManagementLimit(
            handle,
            pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle),
        )

        # Set the memory frequency to be the highest.
        max_mem_freq = max(pynvml.nvmlDeviceGetSupportedMemoryClocks(handle))
        if pynvml.nvmlDeviceGetArchitecture(handle) >= pynvml.NVML_DEVICE_ARCH_AMPERE:
            try:
                pynvml.nvmlDeviceSetMemoryLockedClocks(
                    handle, max_mem_freq, max_mem_freq
                )
            except pynvml.NVMLError_NotSupported:  # type: ignore
                pass

        # Set the SM frequency to be default_freq.
        start_timestamp = time.time_ns()
        if self.default_power_state is not None:
            pynvml.nvmlDeviceSetGpuLockedClocks(
                handle, self.default_power_state, self.default_power_state
            )
            current_freq: int = self.default_power_state
        else:
            pynvml.nvmlDeviceResetGpuLockedClocks(handle)
            current_freq = -1

        events: list[tuple[int, int, int]] = []
        while True:
            target_freq = self._q.get(block=True)
            if target_freq is None:
                break
            end_timestamp = time.time_ns()
            if current_freq != target_freq:
                pynvml.nvmlDeviceSetGpuLockedClocks(handle, target_freq, target_freq)
            next_start_timestamp = time.time_ns()
            if self.output_dir is not None:
                events.append((current_freq, start_timestamp, end_timestamp))
            current_freq = target_freq
            start_timestamp = next_start_timestamp
        end_timestamp = time.time_ns()
        events.append((current_freq, start_timestamp, end_timestamp))

        if pynvml.nvmlDeviceGetArchitecture(handle) >= pynvml.NVML_DEVICE_ARCH_AMPERE:
            try:
                pynvml.nvmlDeviceResetMemoryLockedClocks(handle)
            except pynvml.NVMLError_NotSupported:  # type: ignore
                pass

        pynvml.nvmlDeviceResetGpuLockedClocks(handle)
        pynvml.nvmlShutdown()

        if self.output_dir is None:
            return

        with open(
            f"{self.output_dir}/{self.device_id}.freqevents.csv", "w", encoding="utf-8"
        ) as f:
            f.write("freq,start,end\n")
            for event in events:
                f.write(",".join(map(str, event)) + "\n")

    def _power_limit_manager_process(self) -> None:
        """Receive power limit values through a queue and apply it.

        Records the start/end time of all power limits and outputs events as a CSV file.
        """
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)

        # Reset any frequency settings.
        if pynvml.nvmlDeviceGetArchitecture(handle) >= pynvml.NVML_DEVICE_ARCH_AMPERE:
            try:
                pynvml.nvmlDeviceResetMemoryLockedClocks(handle)
            except pynvml.NVMLError_NotSupported:  # type: ignore
                pass

        pynvml.nvmlDeviceResetGpuLockedClocks(handle)

        # Set the power limit if default is specified.
        start_timestamp = time.time_ns()
        if self.default_power_state is not None:
            pynvml.nvmlDeviceSetPowerManagementLimit(
                handle, self.default_power_state * 1000
            )
            current_pl: int = self.default_power_state
        else:
            default_power_limit = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(
                handle
            )
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, default_power_limit)
            current_pl = pynvml.nvmlDeviceGetPowerManagementLimit(handle) // 1000

        events: list[tuple[int, int, int]] = []
        while True:
            target_pl = self._q.get(block=True)
            if target_pl is None:
                break
            end_timestamp = time.time_ns()
            if current_pl != target_pl:
                pynvml.nvmlDeviceSetPowerManagementLimit(handle, target_pl * 1000)
            next_start_timestamp = time.time_ns()
            events.append((current_pl, start_timestamp, end_timestamp))
            current_pl = target_pl
            start_timestamp = next_start_timestamp
        end_timestamp = time.time_ns()
        events.append((current_pl, start_timestamp, end_timestamp))

        if self.output_dir is None:
            return

        with open(
            f"{self.output_dir}/{self.device_id}.plevents.csv", "w", encoding="utf-8"
        ) as f:
            f.write("pl,start,end\n")
            for event in events:
                f.write(",".join(map(str, event)) + "\n")

        pynvml.nvmlDeviceSetPowerManagementLimit(
            handle,
            pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle),
        )
        pynvml.nvmlShutdown()
