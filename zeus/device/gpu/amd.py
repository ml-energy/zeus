"""AMD GPUs."""

from __future__ import annotations

import functools
import os
import contextlib
import logging
import time
from typing import Sequence
from functools import lru_cache

try:
    # `amdsmi` prints to stdout on import when libamd_smi.so is not found.
    with contextlib.redirect_stdout(None):
        import amdsmi  # type: ignore
# must catch all exceptions, since ImportError is not the only exception that can be raised (ex. OSError on version mismatch).
# Specific exceptions are handled when import and initialization are retested in `amdsmi_is_available`
except Exception:

    class MockAMDSMI:
        """Mock class for AMD SMI library."""

        def __getattr__(self, name):
            """Raise an error if any method is called.

            Since this class is only used when `amdsmi` is not available,
            something has gone wrong if any method is called.
            """
            raise RuntimeError(f"amdsmi is not available and amdsmi.{name} shouldn't have been called. This is a bug.")

    amdsmi = MockAMDSMI()

import zeus.device.gpu.common as gpu_common

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def amdsmi_is_available() -> bool:
    """Check if amdsmi is available."""
    try:
        # `amdsmi` prints to stdout on import when libamd_smi.so is not found.
        with contextlib.redirect_stdout(None):
            import amdsmi  # type: ignore
    except ImportError:
        logger.info("amdsmi is not available.")
        return False
    # usually thrown if amdsmi can't find libamd_smi.so
    except OSError:
        if os.getenv("ROCM_PATH") is None:
            logger.warning("`ROCM_PATH` is not set. Do you have ROCm installed?")
        return False
    # usually thrown if versions of amdsmi and ROCm are incompatible.
    except AttributeError:
        logger.warning(
            "Failed to import amdsmi. Ensure amdsmi's version is at least as high as the current ROCm version."
        )
        return False
    except KeyError as e:
        logger.warning(
            "Failed to import amdsmi due to a key error on: [%s]. Ensure that amdsmi is installed on your system.",
            e,
        )
        return False
    try:
        amdsmi.amdsmi_init()
        logger.info("amdsmi is available and initialized")
        return True
    except amdsmi.AmdSmiLibraryException as e:
        logger.info("amdsmi is available but could not initialize: %s", e)
        return False


def _handle_amdsmi_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except amdsmi.AmdSmiLibraryException as e:
            exception_class = AMDGPU._exception_map.get(e.get_error_code(), gpu_common.ZeusGPUUnknownError)
            raise exception_class(e.get_error_info()) from e

    return wrapper


class AMDGPU(gpu_common.GPU):
    """Implementation of `GPU` for AMD GPUs."""

    def __init__(self, gpu_index: int) -> None:
        """Initialize the GPU object."""
        super().__init__(gpu_index)
        self._get_handle()

        # These values are updated in AMDGPUs constructor
        self._supports_get_total_energy_consumption = True
        self._supports_instant_power_usage = True
        self._supports_average_power_usage = True
        self._is_dual_die_odd_chiplet = False  # Set for MI250/MI250X odd-indexed GPUs

    _exception_map = {
        1: gpu_common.ZeusGPUInvalidArgError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL
        2: gpu_common.ZeusGPUNotSupportedError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
        8: gpu_common.ZeusGPUTimeoutError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_TIMEOUT
        10: gpu_common.ZeusGPUNoPermissionError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM
        15: gpu_common.ZeusGPUMemoryError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_OUT_OF_RESOURCES
        18: gpu_common.ZeusGPUInitError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INIT_ERROR
        31: gpu_common.ZeusGPUNotFoundError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_FOUND
        32: gpu_common.ZeusGPUInitError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT
        34: gpu_common.ZeusGPUDriverNotLoadedError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_DRIVER_NOT_LOADED
        41: gpu_common.ZeusGPUInsufficientSizeError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INSUFFICIENT_SIZE
        45: gpu_common.ZeusGPUDriverNotLoadedError,  # amdsmi.amdsmi_wrapper.AMDSMI_NO_ENERGY_DRV
        46: gpu_common.ZeusGPUDriverNotLoadedError,  # amdsmi.amdsmi_wrapper.AMDSMI_NO_MSR_DRV
        47: gpu_common.ZeusGPUDriverNotLoadedError,  # amdsmi.amdsmi_wrapper.AMDSMI_NO_HSMP_DRV
        48: gpu_common.ZeusGPUNotSupportedError,  # amdsmi.amdsmi_wrapper.AMDSMI_NO_HSMP_SUP
        49: gpu_common.ZeusGPUNotSupportedError,  # amdsmi.amdsmi_wrapper.AMDSMI_NO_HSMP_MSG_SUP
        50: gpu_common.ZeusGPUTimeoutError,  # amdsmi.amdsmi_wrapper.AMDSMI_HSMP_TIMEOUT
        51: gpu_common.ZeusGPUDriverNotLoadedError,  # amdsmi.amdsmi_wrapper.AMDSMI_NO_DRV
        52: gpu_common.ZeusGPULibraryNotFoundError,  # amdsmi.amdsmi_wrapper.AMDSMI_FILE_NOT_FOUND
        53: gpu_common.ZeusGPUInvalidArgError,  # amdsmi.amdsmi_wrapper.AMDSMI_ARG_PTR_NULL
        4294967295: gpu_common.ZeusGPUUnknownError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_UNKNOWN_ERROR
    }

    @_handle_amdsmi_errors
    def _get_handle(self):
        handles = amdsmi.amdsmi_get_processor_handles()
        if len(handles) <= self.gpu_index:
            raise gpu_common.ZeusGPUNotFoundError(
                f"GPU with index {self.gpu_index} not found. Found {len(handles)} GPUs."
            )
        self.handle = amdsmi.amdsmi_get_processor_handles()[self.gpu_index]

    @_handle_amdsmi_errors
    def get_name(self) -> str:
        """Return the name of the GPU model."""
        info = amdsmi.amdsmi_get_gpu_asic_info(self.handle)
        return info["market_name"]

    @property
    def supports_nonblocking_setters(self) -> bool:
        """Return True if the GPU object supports non-blocking configuration setters."""
        return False

    @_handle_amdsmi_errors
    def get_power_management_limit_constraints(self) -> tuple[int, int]:
        """Return the minimum and maximum power management limits. Units: mW."""
        info = amdsmi.amdsmi_get_power_cap_info(self.handle)  # Returns in W
        return (info["min_power_cap"] * 1000, info["max_power_cap"] * 1000)

    @_handle_amdsmi_errors
    def get_power_management_limit(self) -> int:
        """Return the current power management limit. Units: mW."""
        info = amdsmi.amdsmi_get_power_info(self.handle)  # Returns in W
        return int(info["power_limit"]) * 1000

    @_handle_amdsmi_errors
    def set_power_management_limit(self, power_limit_mw: int, block: bool = True) -> None:
        """Set the GPU's power management limit. Unit: mW."""
        current_limit = self.get_power_management_limit()
        if current_limit == power_limit_mw:
            return

        # Units for set_power_cap is microwatts
        amdsmi.amdsmi_set_power_cap(self.handle, 0, int(power_limit_mw * 1000))

    @_handle_amdsmi_errors
    def reset_power_management_limit(self, block: bool = True) -> None:
        """Reset the GPU's power management limit to the default value."""
        info = amdsmi.amdsmi_get_power_cap_info(self.handle)  # Returns in W
        amdsmi.amdsmi_set_power_cap(
            self.handle, 0, cap=int(info["default_power_cap"] * 1e6)
        )  # expects value in microwatts

    @_handle_amdsmi_errors
    def set_persistence_mode(self, enabled: bool, block: bool = True) -> None:
        """Set persistence mode."""
        raise gpu_common.ZeusGPUNotSupportedError("Persistence mode is not supported on AMD GPUs.")

    @_handle_amdsmi_errors
    def get_supported_memory_clocks(self) -> list[int]:
        """Return a list of supported memory clock frequencies. Units: MHz."""
        info = amdsmi.amdsmi_get_clock_info(self.handle, amdsmi.AmdSmiClkType.MEM)  # returns MHz
        return [info["max_clk"], info["min_clk"]]

    @_handle_amdsmi_errors
    def set_memory_locked_clocks(self, min_clock_mhz: int, max_clock_mhz: int, block: bool = True) -> None:
        """Lock the memory clock to a specified range. Units: MHz."""
        amdsmi.amdsmi_set_gpu_clk_range(
            self.handle,
            min_clock_mhz,
            max_clock_mhz,
            clk_type=amdsmi.AmdSmiClkType.MEM,
        )

    @_handle_amdsmi_errors
    def reset_memory_locked_clocks(self, block: bool = True) -> None:
        """Reset the locked memory clocks to the default."""
        # Get default MEM clock values
        info = amdsmi.amdsmi_get_clock_info(self.handle, amdsmi.AmdSmiClkType.MEM)  # returns MHz

        amdsmi.amdsmi_set_gpu_clk_range(
            self.handle,
            info["min_clk"],
            info["max_clk"],
            clk_type=amdsmi.AmdSmiClkType.MEM,
        )  # expects MHz

    @_handle_amdsmi_errors
    def get_supported_graphics_clocks(self, memory_clock_mhz: int | None = None) -> list[int]:
        """Return a list of supported graphics clock frequencies. Units: MHz.

        Args:
            memory_clock_mhz: Memory clock frequency to use. Some GPUs have
                different supported graphics clocks depending on the memory clock.
        """
        pass
        info = amdsmi.amdsmi_get_clock_info(self.handle, amdsmi.AmdSmiClkType.GFX)  # returns MHz
        return [info["max_clk"], info["min_clk"]]

    @_handle_amdsmi_errors
    def set_gpu_locked_clocks(self, min_clock_mhz: int, max_clock_mhz: int, block: bool = True) -> None:
        """Lock the GPU clock to a specified range. Units: MHz."""
        amdsmi.amdsmi_set_gpu_clk_range(
            self.handle,
            min_clock_mhz,
            max_clock_mhz,
            clk_type=amdsmi.AmdSmiClkType.GFX,
        )

    @_handle_amdsmi_errors
    def reset_gpu_locked_clocks(self, block: bool = True) -> None:
        """Reset the locked GPU clocks to the default."""
        # Get default GPU clock values
        info = amdsmi.amdsmi_get_clock_info(self.handle, amdsmi.AmdSmiClkType.GFX)  # returns MHz

        amdsmi.amdsmi_set_gpu_clk_range(
            self.handle,
            info["min_clk"],
            info["max_clk"],
            clk_type=amdsmi.AmdSmiClkType.GFX,
        )  # expects MHz

    def _ensure_not_dual_die_odd_chiplet(self) -> None:
        """Raise an error if the GPU is a chiplet of a dual-die AMD Instinct MI250/MI250X GPU."""
        if self._is_dual_die_odd_chiplet:
            raise gpu_common.ZeusGPUNotSupportedError(
                f"GPU {self.gpu_index} is a chiplet of a dual-die AMD Instinct MI250/MI250X GPU "
                f"that does not support individual power monitoring. AMD's driver only reports power "
                f"for GPU {self.gpu_index - 1}, which represents the COMBINED power draw of BOTH chiplets "
                f"(GPU {self.gpu_index - 1} and GPU {self.gpu_index}).\n\n"
                f"To measure power/energy for workloads on this GPU:\n"
                f"  1. Use GPU {self.gpu_index - 1} for measurements (e.g., PowerMonitor(gpu_indices=[{self.gpu_index - 1}]))\n"
                f"  2. Be aware that measurements include BOTH chiplets and cannot be separated\n"
                f"  3. If you run workloads on GPU {self.gpu_index}, its power consumption will be "
                f"included in GPU {self.gpu_index - 1}'s readings"
            )

    @_handle_amdsmi_errors
    def get_average_power_usage(self) -> int:
        """Return the average power draw of the GPU. Units: mW."""
        self._ensure_not_dual_die_odd_chiplet()

        if not self._supports_average_power_usage:
            raise gpu_common.ZeusGPUNotSupportedError(
                "Average power usage is not supported on this AMD GPU. "
                "This is because amdsmi.amdsmi_get_power_info does not return a valid 'average_socket_power'. "
                "Please use `get_instant_power_usage` instead."
            )
        # returns in W, convert to mW
        power_info = amdsmi.amdsmi_get_power_info(self.handle)
        avg_power = power_info["average_socket_power"]
        if not isinstance(avg_power, int):
            raise gpu_common.ZeusGPUNotSupportedError(
                f"Average power usage is not supported on this AMD GPU. "
                f"amdsmi.amdsmi_get_power_info returned '{avg_power}' for 'average_socket_power'. "
                f"Please use `get_instant_power_usage` instead."
            )
        return avg_power * 1000

    @_handle_amdsmi_errors
    def get_instant_power_usage(self) -> int:
        """Return the current power draw of the GPU. Units: mW."""
        self._ensure_not_dual_die_odd_chiplet()

        if not self._supports_instant_power_usage:
            raise gpu_common.ZeusGPUNotSupportedError(
                "Instant power usage is not supported on this AMD GPU. "
                "This is because amdsmi.amdsmi_get_power_info does not return a valid 'current_socket_power'. "
                "Please use `get_average_power_usage` instead."
            )
        # returns in W, convert to mW
        return int(amdsmi.amdsmi_get_power_info(self.handle)["current_socket_power"]) * 1000

    @_handle_amdsmi_errors
    def get_average_memory_power_usage(self) -> int:
        """Return the average power usage of the GPU's memory. Units: mW."""
        raise gpu_common.ZeusGPUNotSupportedError("Average memory power usage is not supported on AMD GPUs.")

    @_handle_amdsmi_errors
    def supports_get_total_energy_consumption(self) -> bool:
        """Check if the GPU supports retrieving total energy consumption. Returns a future object of the result."""
        return self._supports_get_total_energy_consumption

    @_handle_amdsmi_errors
    def get_total_energy_consumption(self) -> int:
        """Return the total energy consumption of the GPU since driver load. Units: mJ."""
        if not self._supports_get_total_energy_consumption:
            raise gpu_common.ZeusGPUNotSupportedError(
                "Total energy consumption is not supported on this AMD GPU. "
                "This is because the result of `amdsmi.amdsmi_get_energy_count` is not accurate. "
                "Please use `get_average_power_usage` or `get_instant_power_usage` to calculate energy usage."
            )
        energy_dict = amdsmi.amdsmi_get_energy_count(self.handle)
        if "energy_accumulator" in energy_dict:  # Changed since amdsmi 6.2.1
            energy = energy_dict["energy_accumulator"] * energy_dict["counter_resolution"]
        else:
            # Old API: assume has key "power". If not, exception will be handled by _handle_amdsmi_errors.
            energy = energy_dict["power"] * energy_dict["counter_resolution"]

        return int(energy / 1e3)  # returns in micro Joules, convert to mili Joules

    @_handle_amdsmi_errors
    def get_gpu_temperature(self) -> int:
        """Return the current GPU temperature. Units: Celsius.

        We use the hotspot temperatue (as opposed to edge) as we believe it to be more representative
        of the GPU core's temperature under load.
        """
        return amdsmi.amdsmi_get_temp_metric(
            self.handle,
            amdsmi.AmdSmiTemperatureType.HOTSPOT,
            amdsmi.AmdSmiTemperatureMetric.CURRENT,
        )


class AMDGPUs(gpu_common.GPUs):
    """AMD GPU Manager object, containing individual AMDGPU objects, abstracting amdsmi calls and handling related exceptions.

    !!! Important
        Currently only ROCm >= 6.1 is supported.

    `HIP_VISIBLE_DEVICES` environment variable is respected if set.
    For example, if there are 4 GPUs on the node and `HIP_VISIBLE_DEVICES=0,2`,
    only GPUs 0 and 2 are instantiated. In this case, to access
    GPU of HIP index 0, use the index 0, and for HIP index 2, use the index 1.

    When `HIP_VISIBLE_DEVICES` is not set but `CUDA_VISIBLE_DEVICES` is set,
    `CUDA_VISIBLE_DEVICES` is honored as if it were `HIP_VISIBLE_DEVICES`.
    """

    def __init__(self, ensure_homogeneous: bool = False) -> None:
        """Initialize AMDSMI and sets up the GPUs.

        Args:
            ensure_homogeneous (bool): If True, ensures that all tracked GPUs have the same name.
        """
        try:
            amdsmi.amdsmi_init()
            self._init_gpus()
            if ensure_homogeneous:
                self._ensure_homogeneous()
        except amdsmi.AmdSmiLibraryException as e:
            exception_class = AMDGPU._exception_map.get(e.get_error_code(), gpu_common.ZeusBaseGPUError)
            raise exception_class(e.get_error_info()) from e

    @property
    def gpus(self) -> Sequence[AMDGPU]:
        """Return a list of AMDGPU objects being tracked."""
        return self._gpus

    def _init_gpus(self) -> None:
        # Must respect `HIP_VISIBLE_DEVICES` (or `CUDA_VISIBLE_DEVICES`) if set
        if (visible_device := os.environ.get("HIP_VISIBLE_DEVICES")) is not None or (
            visible_device := os.environ.get("CUDA_VISIBLE_DEVICES")
        ) is not None:
            if not visible_device:
                raise gpu_common.ZeusGPUInitError(
                    "HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES is set but empty. "
                    "You can use either one for AMD GPUs, but it should either be unset "
                    "or a comma-separated list of GPU indices."
                )
            visible_indices = [int(idx) for idx in visible_device.split(",")]
        else:
            visible_indices = list(range(len(amdsmi.amdsmi_get_processor_handles())))

        # create the number of visible GPUs
        self._gpus = [AMDGPU(gpu_num) for gpu_num in visible_indices]

        # set _supports_instant_power_usage for all GPUs
        # amdsmi.amdsmi_get_power_info["current_socket_power"] returns "N/A" if not supported
        for gpu in self._gpus:
            gpu._supports_instant_power_usage = isinstance(
                amdsmi.amdsmi_get_power_info(gpu.handle)["current_socket_power"],
                int,
            )

        # set _supports_average_power_usage for all GPUs
        # amdsmi.amdsmi_get_power_info["average_socket_power"] returns "N/A" if not supported
        for gpu in self._gpus:
            gpu._supports_average_power_usage = isinstance(
                amdsmi.amdsmi_get_power_info(gpu.handle)["average_socket_power"],
                int,
            )

        # set _supports_get_total_energy_consumption for all GPUs
        wait_time = 0.5  # seconds
        # Try to get power for energy validation, fallback if needed
        powers = []
        for gpu in self._gpus:
            try:
                power = gpu.get_average_power_usage()
            except gpu_common.ZeusGPUNotSupportedError:
                try:
                    power = gpu.get_instant_power_usage()
                except gpu_common.ZeusGPUNotSupportedError:
                    # Neither average nor instant power available, use 0
                    power = 0
            powers.append(power)
        initial_energies = [gpu.get_total_energy_consumption() for gpu in self._gpus]
        time.sleep(wait_time)
        final_energies = [gpu.get_total_energy_consumption() for gpu in self._gpus]
        measured_energies = [final - initial for final, initial in zip(final_energies, initial_energies)]
        expected_energies = [power * wait_time for power in powers]  # energy = power * time

        for gpu, measured_energy, expected_energy in zip(self._gpus, measured_energies, expected_energies):
            # Check for MI250/MI250X dual-die GPUs and warn users about combined power reporting
            gpu_name = gpu.get_name()
            if "MI250" in gpu_name:
                if gpu.gpu_index % 2 == 1 and expected_energy < 0.001:
                    # This is an odd-indexed MI250/MI250X GPU (chiplet without power reporting)
                    gpu._is_dual_die_odd_chiplet = True
                    gpu._supports_get_total_energy_consumption = False
                    gpu._supports_instant_power_usage = False
                    gpu._supports_average_power_usage = False
                    logger.warning(
                        "GPU %d is a chiplet of a dual-die AMD Instinct MI250/MI250X GPU. "
                        "AMD's driver only reports power for GPU %d, which represents the COMBINED "
                        "power draw of BOTH chiplets (GPU %d and GPU %d). "
                        "Power and energy measurements are not available for this GPU individually.",
                        gpu.gpu_index,
                        gpu.gpu_index - 1,
                        gpu.gpu_index - 1,
                        gpu.gpu_index,
                    )
                elif gpu.gpu_index % 2 == 0 and expected_energy >= 0.001:
                    # This is an even-indexed MI250/MI250X GPU (reports combined power)
                    logger.warning(
                        "GPU %d is a dual-die AMD Instinct MI250/MI250X GPU. "
                        "Power and energy measurements for this GPU represent the COMBINED consumption "
                        "of BOTH chiplets (GPU %d and GPU %d). Workloads running on either chiplet "
                        "will be included in these measurements and cannot be separated.",
                        gpu.gpu_index,
                        gpu.gpu_index,
                        gpu.gpu_index + 1,
                    )

            # Check for zero or very small expected_energy to avoid division by zero
            if expected_energy < 0.001:
                # Skip if already handled as MI250 odd chiplet above
                if not gpu._is_dual_die_odd_chiplet:
                    # Generic case: GPU reports zero power (idle or unsupported)
                    gpu._supports_get_total_energy_consumption = False
                    logger.info(
                        "Disabling `get_total_energy_consumption` for device %d. "
                        "Power reading is zero or negligible (expected energy: %.3f mJ), "
                        "so energy counter validation cannot be performed. "
                        "You can still measure energy by polling either `get_instant_power_usage` or `get_average_power_usage` and integrating over time.",
                        gpu.gpu_index,
                        expected_energy,
                    )
            # Loose bound to rule out very obvious counter problems
            elif 0.1 < measured_energy / expected_energy < 10:
                gpu._supports_get_total_energy_consumption = True
            else:
                gpu._supports_get_total_energy_consumption = False
                logger.info(
                    "Disabling `get_total_energy_consumption` for device %d. The result of `amdsmi.amdsmi_get_energy_count` is not accurate. Expected energy: %d mJ, Measured energy: %d mJ. "
                    "This is a known issue with some AMD GPUs, please see https://github.com/ROCm/amdsmi/issues/38 for more information. "
                    "You can still measure energy by polling either `get_instant_power_usage` or `get_average_power_usage` and integrating over time.",
                    gpu.gpu_index,
                    expected_energy,
                    measured_energy,
                )

    def __del__(self) -> None:
        """Shut down AMDSMI."""
        with contextlib.suppress(amdsmi.AmdSmiException):
            amdsmi.amdsmi_shut_down()  # Ignore error on shutdown. Neccessary for proper cleanup and test functionality
