"""AMD GPUs."""

from __future__ import annotations
import functools
import os
import concurrent.futures
import contextlib
import time
from typing import Sequence
from functools import lru_cache

try:
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
            raise RuntimeError(
                f"amdsmi is not available and amdsmi.{name} shouldn't have been called. "
                "This is a bug."
            )

    amdsmi = MockAMDSMI()

import zeus.device.gpu.common as gpu_common
from zeus.utils.logging import get_logger

logger = get_logger(name=__name__)


@lru_cache(maxsize=1)
def amdsmi_is_available() -> bool:
    """Check if amdsmi is available."""
    try:
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
        logger.warning("Do you have the correct version of ROCm and amdsmi installed?")
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
            exception_class = AMDGPU._exception_map.get(
                e.get_error_code(), gpu_common.ZeusGPUUnknownError
            )
            raise exception_class(e.get_error_info()) from e

    return wrapper


class AMDGPU(gpu_common.GPU):
    """Implementation of `GPU` for AMD GPUs."""

    def __init__(self, gpu_index: int) -> None:
        """Initialize the GPU object."""
        super().__init__(gpu_index)
        self._get_handle()

        self._supportsGetTotalEnergyConsumption = (
            None  # Must be set by `supportsGetTotalEnergyConsumption`
        )

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
    def getName(self) -> str:
        """Return the name of the GPU model."""
        info = amdsmi.amdsmi_get_gpu_asic_info(self.handle)
        return info["market_name"]

    @property
    def supports_nonblocking_setters(self) -> bool:
        """Return True if the GPU object supports non-blocking configuration setters."""
        return False

    @_handle_amdsmi_errors
    def getPowerManagementLimitConstraints(self) -> tuple[int, int]:
        """Return the minimum and maximum power management limits. Units: mW."""
        info = amdsmi.amdsmi_get_power_cap_info(self.handle)  # Returns in W
        return (info["min_power_cap"] * 1000, info["max_power_cap"] * 1000)

    @_handle_amdsmi_errors
    def setPowerManagementLimit(self, power_limit_mw: int, _block: bool = True) -> None:
        """Set the GPU's power management limit. Unit: mW."""
        amdsmi.amdsmi_set_power_cap(
            self.handle, 0, int(power_limit_mw * 1000)
        )  # Units for set_power_cap: microwatts

    @_handle_amdsmi_errors
    def resetPowerManagementLimit(self, _block: bool = True) -> None:
        """Reset the GPU's power management limit to the default value."""
        info = amdsmi.amdsmi_get_power_cap_info(self.handle)  # Returns in W
        amdsmi.amdsmi_set_power_cap(
            self.handle, 0, cap=int(info["default_power_cap"] * 1e6)
        )  # expects value in microwatts

    @_handle_amdsmi_errors
    def setPersistenceMode(self, enabled: bool, _block: bool = True) -> None:
        """Set persistence mode."""
        raise gpu_common.ZeusGPUNotSupportedError(
            "Persistence mode is not supported on AMD GPUs."
        )

    @_handle_amdsmi_errors
    def getSupportedMemoryClocks(self) -> list[int]:
        """Return a list of supported memory clock frequencies. Units: MHz."""
        info = amdsmi.amdsmi_get_clock_info(
            self.handle, amdsmi.AmdSmiClkType.MEM
        )  # returns MHz
        return [info["max_clk"], info["min_clk"]]

    @_handle_amdsmi_errors
    def setMemoryLockedClocks(
        self, min_clock_mhz: int, max_clock_mhz: int, _block: bool = True
    ) -> None:
        """Lock the memory clock to a specified range. Units: MHz."""
        amdsmi.amdsmi_set_gpu_clk_range(
            self.handle,
            min_clock_mhz,
            max_clock_mhz,
            clk_type=amdsmi.AmdSmiClkType.MEM,
        )

    @_handle_amdsmi_errors
    def resetMemoryLockedClocks(self, _block: bool = True) -> None:
        """Reset the locked memory clocks to the default."""
        # Get default MEM clock values
        info = amdsmi.amdsmi_get_clock_info(
            self.handle, amdsmi.AmdSmiClkType.MEM
        )  # returns MHz

        amdsmi.amdsmi_set_gpu_clk_range(
            self.handle,
            info["min_clk"],
            info["max_clk"],
            clk_type=amdsmi.AmdSmiClkType.MEM,
        )  # expects MHz

    @_handle_amdsmi_errors
    def getSupportedGraphicsClocks(
        self, memory_clock_mhz: int | None = None
    ) -> list[int]:
        """Return a list of supported graphics clock frequencies. Units: MHz.

        Args:
            memory_clock_mhz: Memory clock frequency to use. Some GPUs have
                different supported graphics clocks depending on the memory clock.
        """
        pass
        info = amdsmi.amdsmi_get_clock_info(
            self.handle, amdsmi.AmdSmiClkType.GFX
        )  # returns MHz
        return [info["max_clk"], info["min_clk"]]

    @_handle_amdsmi_errors
    def setGpuLockedClocks(
        self, min_clock_mhz: int, max_clock_mhz: int, _block: bool = True
    ) -> None:
        """Lock the GPU clock to a specified range. Units: MHz."""
        amdsmi.amdsmi_set_gpu_clk_range(
            self.handle,
            min_clock_mhz,
            max_clock_mhz,
            clk_type=amdsmi.AmdSmiClkType.GFX,
        )

    @_handle_amdsmi_errors
    def resetGpuLockedClocks(self, _block: bool = True) -> None:
        """Reset the locked GPU clocks to the default."""
        # Get default GPU clock values
        info = amdsmi.amdsmi_get_clock_info(
            self.handle, amdsmi.AmdSmiClkType.GFX
        )  # returns MHz

        amdsmi.amdsmi_set_gpu_clk_range(
            self.handle,
            info["min_clk"],
            info["max_clk"],
            clk_type=amdsmi.AmdSmiClkType.GFX,
        )  # expects MHz

    @_handle_amdsmi_errors
    def getInstantPowerUsage(self) -> int:
        """Return the current power draw of the GPU. Units: mW."""
        # returns in W, convert to mW
        return int(
            int(amdsmi.amdsmi_get_power_info(self.handle)["average_socket_power"])
            * 1000
        )

    @_handle_amdsmi_errors
    def getAverageMemoryPowerUsage(self) -> int:
        """Return the average power usage of the GPU's memory. Units: mW."""
        raise gpu_common.ZeusGPUNotSupportedError(
            "Average memory power usage is not supported on AMD GPUs."
        )

    @_handle_amdsmi_errors
    def supportsGetTotalEnergyConsumption(self) -> bool:
        """Check if the GPU supports retrieving total energy consumption. Returns a future object of the result."""
        # Return cached value if available
        if self._supportsGetTotalEnergyConsumption is not None:
            return self._supportsGetTotalEnergyConsumption

        wait_time = 0.5  # seconds
        threshold = 0.1  # 10% threshold

        power = self.getInstantPowerUsage()
        initial_energy = self.getTotalEnergyConsumption()
        time.sleep(wait_time)
        final_energy = self.getTotalEnergyConsumption()

        measured_energy = final_energy - initial_energy
        expected_energy = power * wait_time  # power is in mW, wait_time is in seconds

        # if the difference between measured and expected energy is less than 1% of the expected energy, then the API is supported
        if abs(measured_energy - expected_energy) < threshold * expected_energy:
            self._supportsGetTotalEnergyConsumption = True
        else:
            self._supportsGetTotalEnergyConsumption = False
            logger.info(
                "Disabling `getTotalEnergyConsumption` for device %d. The result of `amdsmi.amdsmi_get_energy_count` is not accurate. Expected energy: %d mJ, Measured energy: %d mJ"
                "This is a known issue with some AMD GPUs, please see https://github.com/ROCm/amdsmi/issues/38 for more information."
                "Energy metrics will still be available and measured through polling of `getInstantPowerUsage` method.",
                self.gpu_index,
                expected_energy,
                measured_energy,
            )
        return self._supportsGetTotalEnergyConsumption

    @_handle_amdsmi_errors
    def getTotalEnergyConsumption(self) -> int:
        """Return the total energy consumption of the GPU since driver load. Units: mJ."""
        energy_dict = amdsmi.amdsmi_get_energy_count(self.handle)
        if "energy_accumulator" in energy_dict:  # Changed since amdsmi 6.2.1
            energy = (
                energy_dict["energy_accumulator"] * energy_dict["counter_resolution"]
            )
        else:
            # Old API: assume has key "power". If not, exception will be handled by _handle_amdsmi_errors.
            energy = energy_dict["power"] * energy_dict["counter_resolution"]

        return int(energy / 1e3)  # returns in micro Joules, convert to mili Joules


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
            exception_class = AMDGPU._exception_map.get(
                e.get_error_code(), gpu_common.ZeusBaseGPUError
            )
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

        # create a threadpool with the number of visible GPUs
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(visible_indices)
        ) as executor:
            self._gpus = [AMDGPU(gpu_num) for gpu_num in visible_indices]

            # check if GPUs support getTotalEnergyConsumption. Returns a future object of the result.
            futures = [
                executor.submit(gpu.supportsGetTotalEnergyConsumption)
                for gpu in self._gpus
            ]

            # wait for all futures to complete
            concurrent.futures.wait(futures)

    def __del__(self) -> None:
        """Shut down AMDSMI."""
        with contextlib.suppress(amdsmi.AmdSmiException):
            amdsmi.amdsmi_shut_down()  # Ignore error on shutdown. Neccessary for proper cleanup and test functionality
