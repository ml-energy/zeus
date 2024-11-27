"""Measure the GPU time and energy consumption of a block of code."""

from __future__ import annotations

import os
import warnings
from typing import Literal
from time import time
from pathlib import Path
from dataclasses import dataclass
from functools import cached_property

from zeus.monitor.power import PowerMonitor
from zeus.utils.logging import get_logger
from zeus.utils.framework import sync_execution as sync_execution_fn
from zeus.device import get_gpus, get_cpus
from zeus.device.gpu.common import ZeusGPUInitError, EmptyGPUs
from zeus.device.cpu.common import ZeusCPUInitError, ZeusCPUNoPermissionError, EmptyCPUs

logger = get_logger(__name__)


@dataclass
class Measurement:
    """Measurement result of one window.

    Attributes:
        time: Time elapsed (in seconds) during the measurement window.
        gpu_energy: Maps GPU indices to the energy consumed (in Joules) during the
            measurement window. GPU indices are from the DL framework's perspective
            after applying `CUDA_VISIBLE_DEVICES`.
        cpu_energy: Maps CPU indices to the energy consumed (in Joules) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d). This can
            be 'None' if CPU measurement is not available.
        dram_energy: Maps CPU indices to the energy consumed (in Joules) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d)  and DRAM
            measurements are taken from sub-packages within each powerzone. This can be 'None' if
            CPU measurement is not available or DRAM measurement is not available.
    """

    time: float
    gpu_energy: dict[int, float]
    cpu_energy: dict[int, float] | None = None
    dram_energy: dict[int, float] | None = None

    @cached_property
    def total_energy(self) -> float:
        """Total energy consumed (in Joules) during the measurement window."""
        # TODO: Update method to total_gpu_energy, which may cause breaking changes in the examples/
        return sum(self.gpu_energy.values())


@dataclass
class MeasurementState:
    """Measurement state to keep track of measurements in start_window.

    Used in ZeusMonitor to map string keys of measurements to this dataclass.

    Attributes:
        time: The beginning timestamp of the measurement window.
        gpu_energy: Maps GPU indices to the energy consumed (in Joules) during the
            measurement window. GPU indices are from the DL framework's perspective
            after applying `CUDA_VISIBLE_DEVICES`.
        cpu_energy: Maps CPU indices to the energy consumed (in Joules) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d). This can
            be 'None' if CPU measurement is not available.
        dram_energy: Maps CPU indices to the energy consumed (in Joules) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d)  and DRAM
            measurements are taken from sub-packages within each powerzone. This can be 'None' if
            CPU measurement is not available or DRAM measurement is not available.
    """

    time: float
    gpu_energy: dict[int, float]
    cpu_energy: dict[int, float] | None = None
    dram_energy: dict[int, float] | None = None

    @cached_property
    def total_energy(self) -> float:
        """Total energy consumed (in Joules) during the measurement window."""
        return sum(self.gpu_energy.values())


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

    !!! Warning
        Since the monitor may spawn a process to poll the power API on GPUs older than
        Volta, **the monitor should not be instantiated as a global variable
        without guarding it with `if __name__ == "__main__"`**.
        Refer to the "Safe importing of main module" section in the
        [Python documentation](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods)
        for more details.

    ## Integration Example

    ```python
    import time
    from zeus.monitor import ZeusMonitor

    def training():
        # A dummy training function
        time.sleep(5)

    if __name__ == "__main__":
        # Time/Energy measurements for four GPUs will begin and end at the same time.
        gpu_indices = [0, 1, 2, 3]
        monitor = ZeusMonitor(gpu_indices)

        # Mark the beginning of a measurement window. You can use any string
        # as the window name, but make sure it's unique.
        monitor.begin_window("entire_training")

        # Actual work
        training()

        # Mark the end of a measurement window and retrieve the measurment result.
        result = monitor.end_window("entire_training")

        # Print the measurement result.
        print(f"Training consumed {result.total_energy} Joules.")
        for gpu_idx, gpu_energy in result.gpu_energy.items():
            print(f"GPU {gpu_idx} consumed {gpu_energy} Joules.")
    ```

    Attributes:
        gpu_indices (`list[int]`): Indices of all the CUDA devices to monitor, from the
            DL framework's perspective after applying `CUDA_VISIBLE_DEVICES`.
    """

    def __init__(
        self,
        gpu_indices: list[int] | None = None,
        cpu_indices: list[int] | None = None,
        approx_instant_energy: bool = False,
        log_file: str | Path | None = None,
        sync_execution_with: Literal["torch", "jax"] = "torch",
    ) -> None:
        """Instantiate the monitor.

        Args:
            gpu_indices: Indices of all the CUDA devices to monitor. Time/Energy measurements
                will begin and end at the same time for these GPUs (i.e., synchronized).
                If None, all the GPUs available will be used. `CUDA_VISIBLE_DEVICES`
                is respected if set, e.g., GPU index `1` passed into `gpu_indices` when
                `CUDA_VISIBLE_DEVICES=2,3` will be interpreted as CUDA device `3`.
                `CUDA_VISIBLE_DEVICES`s formatted with comma-separated indices are supported.
            cpu_indices: Indices of the CPU packages to monitor. If None, all CPU packages will
                be used.
            approx_instant_energy: When the execution time of a measurement window is
                shorter than the NVML energy counter's update period, energy consumption may
                be observed as zero. In this case, if `approx_instant_energy` is True, the
                window's energy consumption will be approximated by multiplying the current
                instantaneous power consumption with the window's execution time. This should
                be a better estimate than zero, but it's still an approximation.
            log_file: Path to the log CSV file. If `None`, logging will be disabled.
            sync_execution_with: Deep learning framework to use to synchronize CPU/GPU computations.
                Defaults to `"torch"`, in which case `torch.cuda.synchronize` will be used.
                See [`sync_execution`][zeus.utils.framework.sync_execution] for more details.
        """
        # Save arguments.
        self.approx_instant_energy = approx_instant_energy
        self.sync_with: Literal["torch", "jax"] = sync_execution_with

        # Get GPU instances.
        try:
            self.gpus = get_gpus()
        except ZeusGPUInitError:
            self.gpus = EmptyGPUs()

        # Get CPU instance.
        try:
            self.cpus = get_cpus()
        except ZeusCPUInitError:
            self.cpus = EmptyCPUs()
        except ZeusCPUNoPermissionError as err:
            if cpu_indices:
                raise RuntimeError(
                    "Root privilege is required to read RAPL metrics. See "
                    "https://ml.energy/zeus/getting_started/#system-privileges "
                    "for more information or disable CPU measurement by passing cpu_indices=[] to "
                    "ZeusMonitor"
                ) from err
            self.cpus = EmptyCPUs()

        # Resolve GPU indices. If the user did not specify `gpu_indices`, use all available GPUs.
        self.gpu_indices = (
            gpu_indices if gpu_indices is not None else list(range(len(self.gpus)))
        )

        # Resolve CPU indices. If the user did not specify `cpu_indices`, use all available CPUs.
        self.cpu_indices = (
            cpu_indices if cpu_indices is not None else list(range(len(self.cpus)))
        )

        logger.info("Monitoring GPU indices %s.", self.gpu_indices)
        logger.info("Monitoring CPU indices %s", self.cpu_indices)

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

        # A dictionary that maps the string keys of active measurement windows to
        # the state of the measurement window. Each element in the dictionary is a Measurement State
        # object with:
        #     1) Time elapsed at the beginning of this window.
        #     2) Total energy consumed by each >= Volta GPU at the beginning of
        #        this window (`None` for older GPUs).
        #     3) Total energy consumed by each CPU powerzone at the beginning of this window.
        #        ('None' if CPU measurement is not supported)
        #     4) Total energy consumed by each DRAM in powerzones at the beginning of this window.
        #        ('None' if DRAM measurement is not supported)
        self.measurement_states: dict[str, MeasurementState] = {}

        # Initialize power monitors for older architecture GPUs.
        old_gpu_indices = [
            gpu_index
            for gpu_index in self.gpu_indices
            if not self.gpus.supportsGetTotalEnergyConsumption(gpu_index)
        ]
        if old_gpu_indices:
            self.power_monitor = PowerMonitor(
                gpu_indices=old_gpu_indices, update_period=None
            )
        else:
            self.power_monitor = None

    def _get_instant_power(self) -> tuple[dict[int, float], float]:
        """Measure the power consumption of all GPUs at the current time."""
        power_measurement_start_time: float = time()
        power = {
            i: self.gpus.getInstantPowerUsage(i) / 1000.0 for i in self.gpu_indices
        }
        power_measurement_time = time() - power_measurement_start_time
        return power, power_measurement_time

    def begin_window(self, key: str, sync_execution: bool = True) -> None:
        """Begin a new measurement window.

        Args:
            key: Unique name of the measurement window.
            sync_execution: Whether to wait for asynchronously dispatched computations
                to finish before starting the measurement window. For instance, PyTorch
                and JAX will run GPU computations asynchronously, and waiting them to
                finish is necessary to ensure that the measurement window captures all
                and only the computations dispatched within the window.
        """
        # Make sure the key is unique.
        if key in self.measurement_states:
            raise ValueError(f"Measurement window '{key}' already exists")

        # Synchronize execution (e.g., cudaSynchronize) to freeze at the right time.
        if sync_execution and self.gpu_indices:
            sync_execution_fn(self.gpu_indices, sync_with=self.sync_with)

        # Freeze the start time of the profiling window.
        timestamp: float = time()
        gpu_energy_state: dict[int, float] = {}
        for gpu_index in self.gpu_indices:
            # Query energy directly if the GPU has newer architecture.
            # Otherwise, the Zeus power monitor is running in the background to
            # collect power consumption, so we just need to read the log file later.
            if self.gpus.supportsGetTotalEnergyConsumption(gpu_index):
                gpu_energy_state[gpu_index] = (
                    self.gpus.getTotalEnergyConsumption(gpu_index) / 1000.0
                )

        cpu_energy_state: dict[int, float] = {}
        dram_energy_state: dict[int, float] = {}
        for cpu_index in self.cpu_indices:
            cpu_measurement = self.cpus.getTotalEnergyConsumption(cpu_index) / 1000.0
            cpu_energy_state[cpu_index] = cpu_measurement.cpu_mj
            if cpu_measurement.dram_mj is not None:
                dram_energy_state[cpu_index] = cpu_measurement.dram_mj

        # Add measurement state to dictionary.
        self.measurement_states[key] = MeasurementState(
            time=timestamp,
            gpu_energy=gpu_energy_state,
            cpu_energy=cpu_energy_state or None,
            dram_energy=dram_energy_state or None,
        )
        logger.debug("Measurement window '%s' started.", key)

    def end_window(
        self, key: str, sync_execution: bool = True, cancel: bool = False
    ) -> Measurement:
        """End a measurement window and return the time and energy consumption.

        Args:
            key: Name of an active measurement window.
            sync_execution: Whether to wait for asynchronously dispatched computations
                to finish before starting the measurement window. For instance, PyTorch
                and JAX will run GPU computations asynchronously, and waiting them to
                finish is necessary to ensure that the measurement window captures all
                and only the computations dispatched within the window.
            cancel: Whether to cancel the measurement window. If `True`, the measurement
                window is assumed to be cancelled and discarded. Thus, an empty Measurement
                object will be returned and the measurement window will not be recorded in
                the log file either. `sync_execution` is still respected.
        """
        # Retrieve the start time and energy consumption of this window.
        try:
            measurement_state = self.measurement_states.pop(key)
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

        # Synchronize execution (e.g., cudaSynchronize) to freeze at the right time.
        if sync_execution and self.gpu_indices:
            sync_execution_fn(self.gpu_indices, sync_with=self.sync_with)

        # If the measurement window is cancelled, return an empty Measurement object.
        if cancel:
            logger.debug("Measurement window '%s' cancelled.", key)
            return Measurement(
                time=0.0,
                gpu_energy={gpu: 0.0 for gpu in self.gpu_indices},
                cpu_energy={cpu: 0.0 for cpu in self.cpu_indices},
            )

        end_time: float = time()
        start_time = measurement_state.time
        gpu_start_energy = measurement_state.gpu_energy
        cpu_start_energy = measurement_state.cpu_energy
        dram_start_energy = measurement_state.dram_energy

        time_consumption: float = end_time - start_time
        gpu_energy_consumption: dict[int, float] = {}
        for gpu_index in self.gpu_indices:
            # Query energy directly if the GPU has newer architecture.
            if self.gpus.supportsGetTotalEnergyConsumption(gpu_index):
                end_energy = self.gpus.getTotalEnergyConsumption(gpu_index) / 1000.0
                gpu_energy_consumption[gpu_index] = (
                    end_energy - gpu_start_energy[gpu_index]
                )

        cpu_energy_consumption: dict[int, float] = {}
        dram_energy_consumption: dict[int, float] = {}
        for cpu_index in self.cpu_indices:
            cpu_measurement = self.cpus.getTotalEnergyConsumption(cpu_index) / 1000.0
            if cpu_start_energy is not None:
                cpu_energy_consumption[cpu_index] = (
                    cpu_measurement.cpu_mj - cpu_start_energy[cpu_index]
                )
            if dram_start_energy is not None and cpu_measurement.dram_mj is not None:
                dram_energy_consumption[cpu_index] = (
                    cpu_measurement.dram_mj - dram_start_energy[cpu_index]
                )

        # If there are older GPU architectures, the PowerMonitor will take care of those.
        if self.power_monitor is not None:
            energy = self.power_monitor.get_energy(start_time, end_time)
            # Fallback to the instant power measurement if the PowerMonitor does not
            # have the power samples.
            if energy is None:
                energy = {gpu: 0.0 for gpu in self.power_monitor.gpu_indices}
            gpu_energy_consumption |= energy

        # Approximate energy consumption if the measurement window is too short.
        if self.approx_instant_energy:
            for gpu_index in self.gpu_indices:
                if gpu_energy_consumption[gpu_index] == 0.0:
                    gpu_energy_consumption[gpu_index] = power[gpu_index] * (
                        time_consumption - power_measurement_time
                    )

        # Trigger a warning if energy consumption is zero and approx_instant_energy is not enabled.
        if not self.approx_instant_energy and any(
            energy == 0.0 for energy in gpu_energy_consumption.values()
        ):
            warnings.warn(
                "The energy consumption of one or more GPUs was measured as zero. This means that the time duration of the measurement window was shorter than the GPU's energy counter update period. Consider turning on the `approx_instant_energy` option in `ZeusMonitor`, which approximates the energy consumption of a short time window as instant power draw x window duration.",
                stacklevel=1,
            )

        logger.debug("Measurement window '%s' ended.", key)

        # Add to log file.
        if self.log_file is not None:
            self.log_file.write(
                f"{start_time},{key},{time_consumption},"
                + ",".join(str(gpu_energy_consumption[gpu]) for gpu in self.gpu_indices)
                + "\n"
            )
            self.log_file.flush()

        return Measurement(
            time=time_consumption,
            gpu_energy=gpu_energy_consumption,
            cpu_energy=cpu_energy_consumption or None,
            dram_energy=dram_energy_consumption or None,
        )
