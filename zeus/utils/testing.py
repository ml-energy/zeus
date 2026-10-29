"""Utilities for testing."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Sequence

from zeus.device.cpu.common import CPU, CPUs, CpuDramMeasurement
from zeus.device.gpu.common import GPU, GPUs, ZeusGPUInvalidArgError, ZeusGPUNotSupportedError
from zeus.monitor import Measurement, ZeusMonitor
from zeus.utils.framework import sync_execution as sync_execution_fn
from zeus.utils.zeusd import CpuDramPowerLimits, ZonePowerLimits

logger = logging.getLogger(__name__)


class _EnergyCounter:
    """Integrates a piecewise constant power draw over a clock.

    Power is in mW and time is in seconds, so energy is in mJ.
    """

    def __init__(self, power_mw: int, clock: Callable[[], float]) -> None:
        """Start the counter at zero energy."""
        self._clock = clock
        self._power_mw = power_mw
        self._energy_mj = 0.0
        self._last_time = clock()

    def _accumulate(self) -> None:
        now = self._clock()
        self._energy_mj += self._power_mw * (now - self._last_time)
        self._last_time = now

    @property
    def power_mw(self) -> int:
        """Current power draw. Units: mW."""
        return self._power_mw

    @power_mw.setter
    def power_mw(self, power_mw: int) -> None:
        # Integrate the previous power draw up to now before switching.
        self._accumulate()
        self._power_mw = power_mw

    @property
    def energy_mj(self) -> float:
        """Energy consumed since the counter was created. Units: mJ."""
        self._accumulate()
        return self._energy_mj


def _clock_range(min_clock_mhz: int, max_clock_mhz: int) -> tuple[int, int]:
    """Return the clock range, or raise if the minimum exceeds the maximum."""
    if min_clock_mhz > max_clock_mhz:
        raise ZeusGPUInvalidArgError(f"Minimum clock {min_clock_mhz} MHz exceeds maximum clock {max_clock_mhz} MHz.")
    return (min_clock_mhz, max_clock_mhz)


class MockGPU(GPU):
    """A simulated GPU for tests that run without GPU hardware.

    The GPU draws `power_mw`, capped at its current power management limit, and its
    energy counter integrates that power over `clock`. The power draw and the clock
    are fixed at construction. The other constructor arguments are stored in attributes
    of the same name, which tests can change at any time.

    Setters change the simulated state. `set_power_management_limit` raises
    `ZeusGPUInvalidArgError` for a limit outside `power_limit_constraints_mw`, and
    the clock locking setters raise it when the minimum clock exceeds the maximum.

    Attributes:
        persistence_mode: Whether persistence mode is enabled.
        gpu_locked_clocks_mhz: Locked GPU clock range, or `None` if not locked. Units: MHz.
        memory_locked_clocks_mhz: Locked memory clock range, or `None` if not locked. Units: MHz.
    """

    def __init__(
        self,
        gpu_index: int = 0,
        *,
        name: str = "Mock GPU",
        power_mw: int = 100_000,
        memory_power_mw: int | None = None,
        power_limit_constraints_mw: tuple[int, int] = (100_000, 300_000),
        supported_memory_clocks_mhz: Sequence[int] = (1000,),
        supported_graphics_clocks_mhz: Sequence[int] = (1500, 1200, 900),
        supports_total_energy: bool = True,
        temperature_c: int = 40,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the simulated GPU.

        Args:
            gpu_index: Index of the GPU.
            name: Name of the GPU model.
            power_mw: Power the GPU draws while its power limit allows it. Units: mW.
            memory_power_mw: Average power draw of the GPU memory, or `None` if the GPU
                does not report it. Units: mW.
            power_limit_constraints_mw: Minimum and maximum power management limits.
                The maximum is also the default limit. Units: mW.
            supported_memory_clocks_mhz: Supported memory clock frequencies. Units: MHz.
            supported_graphics_clocks_mhz: Supported graphics clock frequencies. Units: MHz.
            supports_total_energy: Whether the GPU has an energy counter. NVIDIA GPUs older
                than Volta do not, and `ZeusMonitor` polls their power instead.
            temperature_c: GPU temperature. Units: Celsius.
            clock: Returns the current time in seconds. The energy counter integrates power
                over it. Tests that patch the time seen by the code under test can pass the
                same function.
        """
        super().__init__(gpu_index)
        self.name = name
        self.memory_power_mw = memory_power_mw
        self.power_limit_constraints_mw = power_limit_constraints_mw
        self.supported_memory_clocks_mhz = list(supported_memory_clocks_mhz)
        self.supported_graphics_clocks_mhz = list(supported_graphics_clocks_mhz)
        self.supports_total_energy = supports_total_energy
        self.temperature_c = temperature_c
        self.persistence_mode = False
        self.gpu_locked_clocks_mhz: tuple[int, int] | None = None
        self.memory_locked_clocks_mhz: tuple[int, int] | None = None
        self._power_mw = power_mw
        self._power_limit_mw = power_limit_constraints_mw[1]
        self._energy = _EnergyCounter(min(power_mw, self._power_limit_mw), clock)

    @property
    def supports_nonblocking_setters(self) -> bool:
        """Return False, since setters take effect immediately."""
        return False

    def get_name(self) -> str:
        """Return the name of the GPU model."""
        return self.name

    def get_power_management_limit_constraints(self) -> tuple[int, int]:
        """Return the minimum and maximum power management limits. Units: mW."""
        return self.power_limit_constraints_mw

    def get_power_management_limit(self) -> int:
        """Return the current power management limit. Units: mW."""
        return self._power_limit_mw

    def set_power_management_limit(self, power_limit_mw: int, block: bool = True) -> None:
        """Set the GPU's power management limit. Unit: mW."""
        min_mw, max_mw = self.power_limit_constraints_mw
        if not min_mw <= power_limit_mw <= max_mw:
            raise ZeusGPUInvalidArgError(f"Power limit {power_limit_mw} mW is outside [{min_mw}, {max_mw}] mW.")
        self._power_limit_mw = power_limit_mw
        self._energy.power_mw = min(self._power_mw, power_limit_mw)

    def reset_power_management_limit(self, block: bool = True) -> None:
        """Reset the GPU's power management limit to the maximum constraint."""
        self.set_power_management_limit(self.power_limit_constraints_mw[1], block)

    def set_persistence_mode(self, enabled: bool, block: bool = True) -> None:
        """Set persistence mode."""
        self.persistence_mode = enabled

    def get_persistence_mode(self) -> bool:
        """Return whether persistence mode is currently enabled."""
        return self.persistence_mode

    def get_supported_memory_clocks(self) -> list[int]:
        """Return a list of supported memory clock frequencies. Units: MHz."""
        return list(self.supported_memory_clocks_mhz)

    def set_memory_locked_clocks(self, min_clock_mhz: int, max_clock_mhz: int, block: bool = True) -> None:
        """Lock the memory clock to a specified range. Units: MHz."""
        self.memory_locked_clocks_mhz = _clock_range(min_clock_mhz, max_clock_mhz)

    def reset_memory_locked_clocks(self, block: bool = True) -> None:
        """Reset the locked memory clocks to the default."""
        self.memory_locked_clocks_mhz = None

    def get_supported_graphics_clocks(self, memory_clock_mhz: int | None = None) -> list[int]:
        """Return a list of supported graphics clock frequencies. Units: MHz.

        Args:
            memory_clock_mhz: Ignored. The same graphics clocks are supported at every memory clock.
        """
        return list(self.supported_graphics_clocks_mhz)

    def set_gpu_locked_clocks(self, min_clock_mhz: int, max_clock_mhz: int, block: bool = True) -> None:
        """Lock the GPU clock to a specified range. Units: MHz."""
        self.gpu_locked_clocks_mhz = _clock_range(min_clock_mhz, max_clock_mhz)

    def reset_gpu_locked_clocks(self, block: bool = True) -> None:
        """Reset the locked GPU clocks to the default."""
        self.gpu_locked_clocks_mhz = None

    def get_average_power_usage(self) -> int:
        """Return the average power draw of the GPU. Units: mW.

        Power is constant between power limit changes, so this equals the instant power draw.
        """
        return self._energy.power_mw

    def get_instant_power_usage(self) -> int:
        """Return the current power draw of the GPU. Units: mW."""
        return self._energy.power_mw

    def get_average_memory_power_usage(self) -> int:
        """Return the average power draw of the GPU's memory. Units: mW."""
        if self.memory_power_mw is None:
            raise ZeusGPUNotSupportedError("This mock GPU does not report memory power.")
        return self.memory_power_mw

    def supports_get_total_energy_consumption(self) -> bool:
        """Check if the GPU supports retrieving total energy consumption."""
        return self.supports_total_energy

    def get_total_energy_consumption(self) -> int:
        """Return the energy consumed since the GPU object was created. Units: mJ."""
        if not self.supports_total_energy:
            raise ZeusGPUNotSupportedError("This mock GPU has no energy counter.")
        return round(self._energy.energy_mj)

    def get_gpu_temperature(self) -> int:
        """Return the current GPU temperature. Units: Celsius."""
        return self.temperature_c


class MockGPUs(GPUs):
    """A collection of `MockGPU` objects that Zeus can use in place of real GPUs.

    `get_gpus` returns `zeus.device.gpu._gpus` whenever it is set, so patching that
    attribute makes every Zeus component in the current process use the simulated GPUs:

    ```python
    def test_training(mocker):
        gpus = MockGPUs([MockGPU(0), MockGPU(1)])
        mocker.patch("zeus.device.gpu._gpus", gpus)
        ...
    ```

    GPU indices are positions in the list. `CUDA_VISIBLE_DEVICES` is not applied.

    !!! Note
        Child processes do not see the patch. `PowerMonitor` polls power in spawned
        processes, so it cannot read simulated GPUs. `ZeusMonitor` starts a
        `PowerMonitor` for GPUs whose `supports_total_energy` is False.
    """

    def __init__(self, gpus: Sequence[MockGPU], ensure_homogeneous: bool = False) -> None:
        """Track the given simulated GPUs.

        Args:
            gpus: Simulated GPUs, in index order.
            ensure_homogeneous: If True, raise `ZeusGPUHeterogeneousError` unless all GPUs
                have the same name.
        """
        self._gpus = list(gpus)
        if ensure_homogeneous:
            self._ensure_homogeneous()

    @property
    def gpus(self) -> Sequence[MockGPU]:
        """Return the simulated GPUs."""
        return self._gpus

    def __del__(self) -> None:
        """Do nothing, since there is no management library to shut down."""


class MockCPU(CPU):
    """A simulated CPU package for tests that run without RAPL or EMI.

    The package and its DRAM draw constant power, and their energy counters
    integrate that power over `clock`.
    """

    def __init__(
        self,
        cpu_index: int = 0,
        *,
        cpu_power_mw: int = 50_000,
        dram_power_mw: int | None = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the simulated CPU package.

        Args:
            cpu_index: Index of the CPU package.
            cpu_power_mw: Power draw of the CPU package. Units: mW.
            dram_power_mw: Power draw of the DRAM, or `None` if the package has no DRAM
                zone. Units: mW.
            clock: Returns the current time in seconds. The energy counters integrate
                power over it.
        """
        super().__init__(cpu_index)
        self._cpu_energy = _EnergyCounter(cpu_power_mw, clock)
        self._dram_energy = None if dram_power_mw is None else _EnergyCounter(dram_power_mw, clock)

    def get_total_energy_consumption(self) -> CpuDramMeasurement:
        """Return the energy consumed since the CPU object was created. Units: mJ."""
        return CpuDramMeasurement(
            cpu_mj=self._cpu_energy.energy_mj,
            dram_mj=None if self._dram_energy is None else self._dram_energy.energy_mj,
        )

    def supports_get_dram_energy_consumption(self) -> bool:
        """Return True if the package has a DRAM zone."""
        return self._dram_energy is not None

    def get_power_limits(self) -> CpuDramPowerLimits:
        """Return zone power limits that are disabled and have no constraints."""
        return CpuDramPowerLimits(
            cpu=ZonePowerLimits(enabled=False, constraints=[]),
            dram=None if self._dram_energy is None else ZonePowerLimits(enabled=False, constraints=[]),
        )


class MockCPUs(CPUs):
    """A collection of `MockCPU` objects that Zeus can use in place of real CPUs.

    As with [`MockGPUs`][zeus.utils.testing.MockGPUs], patch `zeus.device.cpu._cpus`
    to make `get_cpus` return it.
    """

    def __init__(self, cpus: Sequence[MockCPU]) -> None:
        """Track the given simulated CPU packages.

        Args:
            cpus: Simulated CPU packages, in index order.
        """
        self._cpus = list(cpus)

    @property
    def cpus(self) -> Sequence[MockCPU]:
        """Return the simulated CPU packages."""
        return self._cpus

    def __del__(self) -> None:
        """Do nothing, since there is no monitoring interface to shut down."""


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
        ignore_sync_execution: bool = False,
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
            ignore_sync_execution: Whether to ignore `sync_execution` calls. (Default: `False`)
            match_window_name: Whether to make sure window names match. (Default: `True`)
        """
        if log_file is None:
            raise ValueError("`log_file` cannot be `None` for `ReplayZeusMonitor`.")

        self.approx_instant_energy = approx_instant_energy
        self.log_file = open(log_file)
        self.ignore_sync_execution = ignore_sync_execution
        self.match_window_name = match_window_name

        # Infer GPU indices from the log file if not provided.
        header = self.log_file.readline()
        if gpu_indices is None:
            gpu_indices = [int(gpu.split("_")[0][3:]) for gpu in header.split(",")[3:] if gpu]
        self.nvml_gpu_indices = self.gpu_indices = gpu_indices

        logger.info("Replaying from '%s' with GPU indices %s", log_file, gpu_indices)

        # Keep track of ongoing measurement windows.
        self.ongoing_windows = []

    def begin_window(self, key: str, sync_execution: bool = True, restart: bool = False) -> None:
        """Begin a new window.

        This method just pushes the key into a list of ongoing measurement windows,
        and just makes sure it's unique.

        Args:
            key: Name of the measurement window.
            sync_execution: Whether to synchronize CUDA before starting the measurement window.
                (Default: `True`)
            restart: If True and the window already exists, cancel the existing window
                and start a new one.
        """
        if key in self.ongoing_windows:
            if not restart:
                raise RuntimeError(f"Window {key} is already ongoing.")
            self.ongoing_windows.remove(key)
        self.ongoing_windows.append(key)

        if not self.ignore_sync_execution and sync_execution:
            sync_execution_fn(self.gpu_indices)

        logger.info("Measurement window '%s' started.", key)

    def end_window(self, key: str, sync_execution: bool = True, cancel: bool = False) -> Measurement:
        """End an ongoing window.

        This method pops the key from a list of ongoing measurement windows and
        constructs a `Measurement` object corresponding to the name of the window
        from the log file. If the name of the window does not match the expected
        one, a `RuntimeError` is raised.

        Args:
            key: Name of the measurement window.
            sync_execution: Whether to synchronize CUDA before ending the measurement window.
                (Default: `True`)
            cancel: Whether to cancel the measurement window. This will not consume a
                line from the log file. (Default: `False`)
        """
        try:
            self.ongoing_windows.remove(key)
        except ValueError:
            raise RuntimeError(f"Window {key} is not ongoing.") from None

        if not self.ignore_sync_execution and sync_execution:
            sync_execution_fn(self.gpu_indices)

        if cancel:
            logger.info("Measurement window '%s' cancelled.", key)
            return Measurement(
                time=0.0,
                gpu_energy={gpu_index: 0.0 for gpu_index in self.gpu_indices},
            )

        # Read the next line from the log file.
        assert self.log_file is not None
        line = self.log_file.readline()
        if not line:
            raise RuntimeError("No more lines in the log file.")
        _, window_name, *nums = line.split(",")
        if self.match_window_name and window_name != key:
            raise RuntimeError(f"Was expecting {window_name}, not {key}.")
        if len(nums) != len(self.gpu_indices) + 1:
            raise RuntimeError(f"Line has unexpected number of energy measurements: {line}")
        time_consumption, *energy_consumptions = map(float, nums)
        energy = dict(zip(self.gpu_indices, energy_consumptions))
        measurement = Measurement(time=time_consumption, gpu_energy=energy)

        logger.info("Measurement window '%s' ended (%s).", key, measurement)

        return measurement
