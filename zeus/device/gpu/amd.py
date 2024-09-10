"""AMD GPUs."""

from __future__ import annotations
import functools
import os
import contextlib
from typing import Sequence
from functools import lru_cache

try:
    import amdsmi  # type: ignore
except ImportError:

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
        # XXX(Jae-Won): Right now, the energy API's unit is broken (either the
        # `power` field or the `counter_resolution` field). Before that, we're
        # disabling the energy API.
        self._supportsGetTotalEnergyConsumption = False

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
            amdsmi.amdsmi_get_power_info(self.handle)["average_socket_power"] * 1000
        )

    @_handle_amdsmi_errors
    def getAverageMemoryPowerUsage(self) -> int:
        """Return the average power usage of the GPU's memory. Units: mW."""
        raise gpu_common.ZeusGPUNotSupportedError(
            "Average memory power usage is not supported on AMD GPUs."
        )

    @_handle_amdsmi_errors
    def supportsGetTotalEnergyConsumption(self) -> bool:
        """Check if the GPU supports retrieving total energy consumption."""
        if self._supportsGetTotalEnergyConsumption is None:
            try:
                _ = amdsmi.amdsmi_get_energy_count(self.handle)
                self._supportsGetTotalEnergyConsumption = True
            except amdsmi.AmdSmiLibraryException as e:
                if (
                    e.get_error_code() == 2
                ):  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
                    self._supportsGetTotalEnergyConsumption = False
                else:
                    raise e

        return self._supportsGetTotalEnergyConsumption

    @_handle_amdsmi_errors
    def getTotalEnergyConsumption(self) -> int:
        """Return the total energy consumption of the GPU since driver load. Units: mJ."""
        info = amdsmi.amdsmi_get_energy_count(self.handle)
        return int(
            info["power"] / 1e3
        )  # returns in micro Joules, convert to mili Joules


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
        except amdsmi.AmdSmiException as e:
            exception_class = AMDGPU._exception_map.get(
                e.value, gpu_common.ZeusBaseGPUError
            )
            raise exception_class(e.msg) from e

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

        self._gpus = [AMDGPU(gpu_num) for gpu_num in visible_indices]

    def __del__(self) -> None:
        """Shut down AMDSMI."""
        with contextlib.suppress(amdsmi.AmdSmiException):
            amdsmi.amdsmi_shut_down()  # Ignore error on shutdown. Neccessary for proper cleanup and test functionality
