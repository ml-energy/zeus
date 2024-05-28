"""NVIDIA GPUs."""

from __future__ import annotations

import functools
import os
import contextlib
from typing import Sequence

import pynvml

import zeus.device.gpu.common as gpu_common
from zeus.utils.logging import get_logger

logger = get_logger(name=__name__)


def nvml_is_available() -> bool:
    """Check if NVML is available."""
    try:
        import pynvml
    except ImportError:
        logger.info(
            "Failed to import `pynvml`. Make sure you have package `nvidia-ml-py` installed."
        )
        return False

    # Detect unofficial pynvml packages.
    # If detected, this should be a critical error.
    if not hasattr(pynvml, "_nvmlGetFunctionPointer"):
        logger.error("Unoffical pynvml package detected!")
        raise ImportError(
            "Unofficial pynvml package detected! "
            "This causes conflicts with the official NVIDIA bindings. "
            "Please remove with `pip uninstall pynvml` and instead use the official "
            "bindings from NVIDIA: `nvidia-ml-py`. "
        )

    try:
        pynvml.nvmlInit()
        logger.info("pynvml is available and initialized.")
        return True
    except pynvml.NVMLError:
        logger.info("pynvml is available but could not initialize.")
        return False


def _handle_nvml_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except pynvml.NVMLError as e:
            exception_class = NVIDIAGPU._exception_map.get(
                e.value,  # pyright: ignore[reportAttributeAccessIssue]
                gpu_common.ZeusGPUUnknownError,
            )
            raise exception_class(str(e)) from e

    return wrapper


class NVIDIAGPU(gpu_common.GPU):
    """Control a Single NVIDIA GPU.

    Uses NVML Library to control and query GPU. There is a 1:1 mapping between the methods in this class and the NVML library functions.
    Zeus GPU Exceptions are raised when NVML errors occur.
    To ensure computational efficiency, this class utilizes caching (ex. saves the handle) to avoid repeated calls to NVML.
    """

    def __init__(self, gpu_index: int) -> None:
        """Initializes the NVIDIAGPU object with a specified GPU index. Acquires a handle to the GPU using `pynvml.nvmlDeviceGetHandleByIndex`."""
        super().__init__(gpu_index)
        self._get_handle()
        self._supportsGetTotalEnergyConsumption = None

    _exception_map = {
        pynvml.NVML_ERROR_UNINITIALIZED: gpu_common.ZeusGPUInitError,
        pynvml.NVML_ERROR_INVALID_ARGUMENT: gpu_common.ZeusGPUInvalidArgError,
        pynvml.NVML_ERROR_NOT_SUPPORTED: gpu_common.ZeusGPUNotSupportedError,
        pynvml.NVML_ERROR_NO_PERMISSION: gpu_common.ZeusGPUNoPermissionError,
        pynvml.NVML_ERROR_ALREADY_INITIALIZED: gpu_common.ZeusGPUAlreadyInitializedError,
        pynvml.NVML_ERROR_NOT_FOUND: gpu_common.ZeusGPUNotFoundError,
        pynvml.NVML_ERROR_INSUFFICIENT_SIZE: gpu_common.ZeusGPUInsufficientSizeError,
        pynvml.NVML_ERROR_INSUFFICIENT_POWER: gpu_common.ZeusGPUInsufficientPowerError,
        pynvml.NVML_ERROR_DRIVER_NOT_LOADED: gpu_common.ZeusGPUDriverNotLoadedError,
        pynvml.NVML_ERROR_TIMEOUT: gpu_common.ZeusGPUTimeoutError,
        pynvml.NVML_ERROR_IRQ_ISSUE: gpu_common.ZeusGPUIRQError,
        pynvml.NVML_ERROR_LIBRARY_NOT_FOUND: gpu_common.ZeusGPULibraryNotFoundError,
        pynvml.NVML_ERROR_FUNCTION_NOT_FOUND: gpu_common.ZeusGPUFunctionNotFoundError,
        pynvml.NVML_ERROR_CORRUPTED_INFOROM: gpu_common.ZeusGPUCorruptedInfoROMError,
        pynvml.NVML_ERROR_GPU_IS_LOST: gpu_common.ZeusGPULostError,
        pynvml.NVML_ERROR_RESET_REQUIRED: gpu_common.ZeusGPUResetRequiredError,
        pynvml.NVML_ERROR_OPERATING_SYSTEM: gpu_common.ZeusGPUOperatingSystemError,
        pynvml.NVML_ERROR_LIB_RM_VERSION_MISMATCH: gpu_common.ZeusGPULibRMVersionMismatchError,
        pynvml.NVML_ERROR_MEMORY: gpu_common.ZeusGPUMemoryError,
        pynvml.NVML_ERROR_UNKNOWN: gpu_common.ZeusGPUUnknownError,
    }

    @_handle_nvml_errors
    def _get_handle(self):
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

    @_handle_nvml_errors
    def getPowerManagementLimitConstraints(self) -> tuple[int, int]:
        """Returns the minimum and maximum power management limits for the specified GPU. Units: mW."""
        min_, max_ = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(self.handle)
        return (min_, max_)

    @_handle_nvml_errors
    def setPersistenceMode(self, enable: bool) -> None:
        """If enable = True, enables persistence mode for the specified GPU. If enable = False, disables persistence mode."""
        if enable:
            pynvml.nvmlDeviceSetPersistenceMode(
                self.handle, pynvml.NVML_FEATURE_ENABLED
            )
        else:
            pynvml.nvmlDeviceSetPersistenceMode(
                self.handle, pynvml.NVML_FEATURE_DISABLED
            )

    @_handle_nvml_errors
    def setPowerManagementLimit(self, value: int) -> None:
        """Sets the power management limit for the specified GPU to the given value. Unit: mW."""
        pynvml.nvmlDeviceSetPowerManagementLimit(self.handle, value)

    @_handle_nvml_errors
    def resetPowerManagementLimit(self) -> None:
        """Resets the power management limit for the specified GPU to the default value."""
        pynvml.nvmlDeviceSetPowerManagementLimit(
            self.handle,
            pynvml.nvmlDeviceGetPowerManagementDefaultLimit(self.handle),
        )

    @_handle_nvml_errors
    def setMemoryLockedClocks(self, minMemClockMHz: int, maxMemClockMHz: int) -> None:
        """Locks the memory clock of the specified GPU to a range defined by the minimum and maximum memory clock frequencies.  Units: MHz."""
        pynvml.nvmlDeviceSetMemoryLockedClocks(
            self.handle, minMemClockMHz, maxMemClockMHz
        )

    @_handle_nvml_errors
    def getSupportedMemoryClocks(self) -> list[int]:
        """Returns a list of supported memory clock frequencies for the specified GPU. Units: MHz."""
        return pynvml.nvmlDeviceGetSupportedMemoryClocks(self.handle)

    @_handle_nvml_errors
    def getSupportedGraphicsClocks(self, freq: int) -> list[int]:
        """Returns a list of supported graphics clock frequencies for the specified GPU at a given frequency. Units: MHz."""
        return pynvml.nvmlDeviceGetSupportedGraphicsClocks(self.handle, freq)

    @_handle_nvml_errors
    def getName(self) -> str:
        """Returns the name of the specified GPU."""
        return pynvml.nvmlDeviceGetName(self.handle)

    @_handle_nvml_errors
    def setGpuLockedClocks(self, minGpuClockMHz: int, maxGpuClockMHz: int) -> None:
        """Locks the GPU clock of the specified GPU to a range defined by the minimum and maximum GPU clock frequencies. Units: MHz."""
        pynvml.nvmlDeviceSetGpuLockedClocks(self.handle, minGpuClockMHz, maxGpuClockMHz)

    @_handle_nvml_errors
    def resetMemoryLockedClocks(self) -> None:
        """Resets the memory locked clocks of the specified GPU to their default values."""
        pynvml.nvmlDeviceResetMemoryLockedClocks(self.handle)

    @_handle_nvml_errors
    def resetGpuLockedClocks(self) -> None:
        """Resets the GPU locked clocks of the specified GPU to their default values."""
        pynvml.nvmlDeviceResetGpuLockedClocks(self.handle)

    @_handle_nvml_errors
    def getInstantPowerUsage(self) -> int:
        """Returns the current power usage of the specified GPU. Units: mW."""
        metric = pynvml.nvmlDeviceGetFieldValues(
            self.handle, [pynvml.NVML_FI_DEV_POWER_INSTANT]
        )[0]
        if (ret := metric.nvmlReturn) != pynvml.NVML_SUCCESS:
            raise pynvml.NVMLError(ret)
        return metric.value.siVal

    @_handle_nvml_errors
    def supportsGetTotalEnergyConsumption(self) -> bool:
        """Returns True if the specified GPU supports retrieving the total energy consumption."""
        # NVIDIA GPUs Volta or newer support this method
        if self._supportsGetTotalEnergyConsumption is None:
            self._supportsGetTotalEnergyConsumption = (
                pynvml.nvmlDeviceGetArchitecture(self.handle)
                >= pynvml.NVML_DEVICE_ARCH_VOLTA
            )

        return self._supportsGetTotalEnergyConsumption

    @_handle_nvml_errors
    def getTotalEnergyConsumption(self) -> int:
        """Returns the total energy consumption of the specified GPU. Units: mJ."""
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)


class UnprivilegedNVIDIAGPU(NVIDIAGPU):
    """Control a Single NVIDIA GPU with no SYS_ADMIN privileges.

    Uses NVML Library to control and query GPU. There is a 1:1 mapping between the methods in this class and the NVML library functions.
    Zeus GPU Exceptions are raised when NVML errors occur.
    To ensure computational efficiency, this class utilizes caching (ex. saves the handle) to avoid repeated calls to NVML.
    """

    pass


class NVIDIAGPUs(gpu_common.GPUs):
    """NVIDIA GPU Manager object, containing individual NVIDIAGPU objects, abstracting pyNVML calls and handling related exceptions.

    This class provides a high-level interface to interact with NVIDIA GPUs. `CUDA_VISIBLE_DEVICES` environment variable is respected if set. For example, if there are
    4 GPUs and `CUDA_VISIBLE_DEVICES=0,2`, only GPUs 0 and 2 are instantiated. In this case, to access
    GPU of CUDA index 0, use the index 0, and for CUDA index 2, use the index 1.

    This class provides a 1:1 mapping between the methods and NVML library functions. For example, if you want to do the following:

    ```python
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    constraints = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
    ```

    You can now do:
    ```python
    gpus = get_gpus() # returns a NVIDIAGPUs object
    constraints =  gpus.getPowerManagementLimitConstraints(gpu_index)
    ```

    Note: This class instantiates (grabs the handle, by calling `pynvml.nvmlDeviceGetHandleByIndex`) all GPUs that are visible to the system, as determined by the `CUDA_VISIBLE_DEVICES` environment variable if set.
    """

    def __init__(self, ensure_homogeneous: bool = False) -> None:
        """Instantiates NVIDIAGPUs object, setting up tracking for specified NVIDIA GPUs.

        Args:
            ensure_homogeneous (bool): If True, ensures that all tracked GPUs have the same name (return value of `nvmlDeviceGetName`). False by default.
        """
        try:
            pynvml.nvmlInit()
            self._init_gpus()
            if ensure_homogeneous:
                self._ensure_homogeneous()
        except pynvml.NVMLError as e:
            exception_class = NVIDIAGPU._exception_map.get(
                e.value,  # pyright: ignore[reportAttributeAccessIssue]
                gpu_common.ZeusBaseGPUError,
            )
            raise exception_class(
                e.msg  # pyright: ignore[reportAttributeAccessIssue]
            ) from e

    @property
    def gpus(self) -> Sequence[gpu_common.GPU]:
        """Returns a list of NVIDIAGPU objects being tracked."""
        return self._gpus

    def _init_gpus(self) -> None:
        # Must respect `CUDA_VISIBLE_DEVICES` if set
        if (visible_device := os.environ.get("CUDA_VISIBLE_DEVICES")) is not None:
            self.visible_indices = [int(idx) for idx in visible_device.split(",")]
        else:
            self.visible_indices = list(range(pynvml.nvmlDeviceGetCount()))

        # initialize all GPUs
        self._gpus = [NVIDIAGPU(gpu_num) for gpu_num in self.visible_indices]

    def __del__(self) -> None:
        """Shuts down the NVIDIA GPU monitoring library to release resources and clean up."""
        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlShutdown()
