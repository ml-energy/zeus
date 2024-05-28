"""AMD GPUs."""

from __future__ import annotations
import functools
import os
import contextlib
from typing import Sequence

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
    except amdsmi.AmdSmiLibraryException:
        logger.info("amdsmi is available but could not initialize.")
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
    """Control a Single AMD GPU.

    Uses amdsmi Library to control and query GPU. There is a 1:1 mapping between the methods in this class and the amdsmi library functions.
    Zeus GPU Exceptions are raised when amdsmi errors occur.
    To ensure computational efficiency, this class utilizes caching (ex. saves the handle) to avoid repeated calls to amdsmi.
    """

    def __init__(self, gpu_index: int) -> None:
        """Initializes the AMDGPU object with a specified GPU index. Acquires a handle to the GPU using `amdsmi.amdsmi_get_processor_handles()`."""
        super().__init__(gpu_index)
        self._get_handle()
        self._supportsGetTotalEnergyConsumption = None

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
    def getPowerManagementLimitConstraints(self) -> tuple[int, int]:
        """Returns the minimum and maximum power management limits for the specified GPU. Units: mW."""
        info = amdsmi.amdsmi_get_power_cap_info(self.handle)  # Returns in W
        return (info["min_power_cap"] * 1000, info["max_power_cap"] * 1000)

    @_handle_amdsmi_errors
    def setPersistenceMode(self, enable: bool) -> None:
        """If enable = True, enables persistence mode for the specified GPU. If enable = False, disables persistence mode."""
        # N/A for AMD GPUs.
        pass

    @_handle_amdsmi_errors
    def setPowerManagementLimit(self, value: int) -> None:
        """Sets the power management limit for the specified GPU to the given value. Unit: mW."""
        amdsmi.amdsmi_set_power_cap(
            self.handle, 0, int(value * 1000)
        )  # Units for set_power_cap: microwatts

    @_handle_amdsmi_errors
    def resetPowerManagementLimit(self) -> None:
        """Resets the power management limit for the specified GPU to the default value."""
        info = amdsmi.amdsmi_get_power_cap_info(self.handle)  # Returns in W
        amdsmi.amdsmi_set_power_cap(
            self.handle, 0, cap=int(info["default_power_cap"] * 1e6)
        )  # expects value in microwatts

    @_handle_amdsmi_errors
    def setMemoryLockedClocks(self, minMemClockMHz: int, maxMemClockMHz: int) -> None:
        """Locks the memory clock of the specified GPU to a range defined by the minimum and maximum memory clock frequencies. Units: MHz."""
        amdsmi.amdsmi_set_gpu_clk_range(
            self.handle,
            minMemClockMHz,
            maxMemClockMHz,
            clk_type=amdsmi.AmdSmiClkType.MEM,
        )

    @_handle_amdsmi_errors
    def getSupportedMemoryClocks(self) -> list[int]:
        """Returns a list of supported memory clock frequencies for the specified GPU. Units: MHz."""
        raise gpu_common.ZeusGPUNotSupportedError(
            "AMDSMI does not support querying memory frequencies"
        )

    @_handle_amdsmi_errors
    def getSupportedGraphicsClocks(self, freq: int) -> list[int]:
        """Returns a list of supported graphics clock frequencies for the specified GPU at a given frequency. Units: MHz."""
        raise gpu_common.ZeusGPUNotSupportedError(
            "AMDSMI does not support querying GFX frequencies given a memory frequency"
        )

    @_handle_amdsmi_errors
    def getName(self) -> str:
        """Returns the name of the specified GPU."""
        info = amdsmi.amdsmi_get_gpu_asic_info(self.handle)
        return info["market_name"]

    @_handle_amdsmi_errors
    def setGpuLockedClocks(self, minGpuClockMHz: int, maxGpuClockMHz: int) -> None:
        """Locks the GPU clock of the specified GPU to a range defined by the minimum and maximum GPU clock frequencies.  Units: MHz."""
        amdsmi.amdsmi_set_gpu_clk_range(
            self.handle,
            minGpuClockMHz,
            maxGpuClockMHz,
            clk_type=amdsmi.AmdSmiClkType.GFX,
        )

    @_handle_amdsmi_errors
    def resetMemoryLockedClocks(self) -> None:
        """Resets the memory locked clocks of the specified GPU to their default values."""
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
    def resetGpuLockedClocks(self) -> None:
        """Resets the GPU locked clocks of the specified GPU to their default values."""
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
        """Returns the current power usage of the specified GPU. Units: mW."""
        # returns in W, convert to mW
        return int(
            amdsmi.amdsmi_get_power_info(self.handle)["average_socket_power"] * 1000
        )

    @_handle_amdsmi_errors
    def supportsGetTotalEnergyConsumption(self) -> bool:
        """Returns True if the specified GPU supports retrieving the total energy consumption."""
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
        """Returns the total energy consumption of the specified GPU. Units: mJ."""
        info = amdsmi.amdsmi_get_energy_count(self.handle)
        return int(
            info["power"] / 1e3
        )  # returns in micro Joules, convert to mili Joules


class UnprivilegedAMDGPU(AMDGPU):
    """Control a Single AMD GPU with no SYS_ADMIN privileges.

    Uses amdsmi Library to control and query GPU. There is a 1:1 mapping between the methods in this class and the amdsmi library functions.
    Zeus GPU Exceptions are raised when amdsmi errors occur.
    To ensure computational efficiency, this class utilizes caching (ex. saves the handle) to avoid repeated calls to amdsmi.
    """

    pass


class AMDGPUs(gpu_common.GPUs):
    """AMD GPU Manager object, containing individual AMDGPU objects, abstracting amdsmi calls and handling related exceptions.

    !!! Important
        Currently only ROCM 6.0 is supported.

    This class provides a high-level interface to interact with AMD GPUs. `HIP_VISIBLE_DEVICES` environment variable is respected if set. For example, if there are
    4 GPUs and `HIP_VISIBLE_DEVICES=0,2`, only GPUs 0 and 2 are instantiated. In this case, to access
    GPU of HIP index 0, use the index 0, and for HIP index 2, use the index 1.

    This class provides a 1:1 mapping between the methods and AMDSMI library functions. For example, if you want to do the following:

    ```python
    handle = amdsmi.amdsmi_get_processor_handles()[gpu_index]
    info = amdsmi.amdsmi_get_power_cap_info(self.handle)
    constraints = (info.min_power_cap, info.max_power_cap)
    ```

    You can now do:
    ```python
    gpus = get_gpus() # returns a AMDGPUs object
    constraints =  gpus.getPowerManagementLimitConstraints(gpu_index)
    ```

    Note: This class instantiates (grabs the handle, by calling `amdsmi.amdsmi_get_processor_handles()`) all GPUs that are visible to the system, as determined by the `HIP_VISIBLE_DEVICES` environment variable if set.
    """

    def __init__(self, ensure_homogeneous: bool = False) -> None:
        """Instantiates NVIDIAGPUs object, setting up tracking for specified NVIDIA GPUs.

        Args:
            ensure_homogeneous (bool, optional): If True, ensures that all tracked GPUs have the same name (return value of amdsmi.amdsmi_get_gpu_asic_info(handle).market_name). False by default.
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
    def gpus(self) -> Sequence[gpu_common.GPU]:
        """Returns a list of AMDGPU objects being tracked."""
        return self._gpus

    def _init_gpus(self) -> None:
        # Must respect `HIP_VISIBLE_DEVICES` if set
        if (visible_device := os.environ.get("HIP_VISIBLE_DEVICES")) is not None:
            self.visible_indices = [int(idx) for idx in visible_device.split(",")]
        else:
            self.visible_indices = list(
                range(len(amdsmi.amdsmi_get_processor_handles()))
            )

        self._gpus = [AMDGPU(gpu_num) for gpu_num in self.visible_indices]

    def __del__(self) -> None:
        """Shuts down the AMD GPU monitoring library to release resources and clean up."""
        with contextlib.suppress(amdsmi.AmdSmiException):
            amdsmi.amdsmi_shut_down()  # Ignore error on shutdown. Neccessary for proper cleanup and test functionality
