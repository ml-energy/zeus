"""GPU management module for Zeus."""

from __future__ import annotations
import abc
import functools
import os
from typing import TYPE_CHECKING
import contextlib

import pynvml  # necessary for testing to mock!

from zeus.device.exception import ZeusBaseGPUError
from zeus.util import pynvml_is_available, amdsmi_is_available

if TYPE_CHECKING:
    import amdsmi

""" EXCEPTION WRAPPERS """


class ZeusGPUInitError(ZeusBaseGPUError):
    """Import error or GPU library initialization failures."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUInvalidArgError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Invalid Argument."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUNotSupportedError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Not Supported Operation on GPU."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUNoPermissionError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for No Permission to perform GPU operation."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUAlreadyInitializedError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Already Initialized GPU."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUNotFoundError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Not Found GPU."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUInsufficientSizeError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Insufficient Size."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUInsufficientPowerError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Insufficient Power."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUDriverError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Driver Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUTimeoutError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Timeout Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUIRQError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for IRQ Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPULibraryNotFoundError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Library Not Found Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUFunctionNotFoundError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Function Not Found Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUCorruptedInfoROMError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Corrupted Info ROM Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPULostError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Lost GPU Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUResetRequiredError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Reset Required Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUOperatingSystemError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Operating System Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPULibRMVersionMismatchError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for LibRM Version Mismatch Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUMemoryError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Insufficient Memory Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUUnknownError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Unknown Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


""" SINGLE GPU MANAGEMENT OBJECTS """


class GPU(abc.ABC):
    """Abstract base class for GPU management.

    This class defines the interface for interacting with GPUs, subclasses should implement the methods to interact with specific GPU libraries
    (e.g., NVML for NVIDIA GPUs).
    """

    def __init__(self, gpu_index: int) -> None:
        """Initialize the GPU with a specified index."""
        self.gpu_index = gpu_index

    @abc.abstractmethod
    def getPowerManagementLimitConstraints(self) -> tuple[int, int]:
        """Return the minimum and maximum power management limits for the GPU. Units: mW."""
        pass

    @abc.abstractmethod
    def setPersistenceMode(self, enable: bool) -> None:
        """Enable persistence mode for the GPU."""
        pass

    @abc.abstractmethod
    def setPowerManagementLimit(self, value: int = None) -> None:
        """Set the power management limit for the GPU to a specified value or default. Unit: mW."""
        pass

    @abc.abstractmethod
    def resetPowerManagementLimit(self) -> None:
        """Resets the power management limit for the specified GPU to the default value."""
        pass

    @abc.abstractmethod
    def setMemoryLockedClocks(self, minMemClockMHz: int, maxMemClockMHz: int) -> None:
        """Lock the memory clock to a specified range. Units: MHz."""
        pass

    @abc.abstractmethod
    def getSupportedMemoryClocks(self) -> list[int]:
        """Return a list of supported memory clock frequencies for the GPU. Units: MHz."""
        pass

    @abc.abstractmethod
    def getSupportedGraphicsClocks(self, freq: int) -> list[int]:
        """Return a list of supported graphics clock frequencies for a given memory frequency. Units: MHz."""
        pass

    @abc.abstractmethod
    def getName(self) -> str:
        """Return the name of the GPU."""
        pass

    @abc.abstractmethod
    def setGpuLockedClocks(
        self, index: int, minMemClockMHz: int, maxMemClockMHz: int
    ) -> None:
        """Lock the GPU clock to a specified range. Units: MHz."""
        pass

    @abc.abstractmethod
    def resetMemoryLockedClocks(self) -> None:
        """Reset the memory locked clocks to default values."""
        pass

    @abc.abstractmethod
    def resetGpuLockedClocks(self) -> None:
        """Reset the GPU locked clocks to default values."""
        pass

    @abc.abstractmethod
    def getPowerUsage(self) -> int:
        """Return the current power usage of the GPU. Units: mW."""
        pass

    @abc.abstractmethod
    def supportsGetTotalEnergyConsumption(self) -> bool:
        """Check if the GPU supports retrieving total energy consumption."""
        pass

    @abc.abstractmethod
    def getTotalEnergyConsumption(self) -> int:
        """Return the total energy consumption of the GPU since driver load. Units: mJ."""
        pass


""" GPU MANAGEMENT OBJECTS """


def _handle_nvml_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except pynvml.NVMLError as e:
            exception_class = NVIDIAGPU._exception_map.get(e.value, ZeusGPUUnknownError)
            raise exception_class(e.msg) from e

    return wrapper


class NVIDIAGPU(GPU):
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
        pynvml.NVML_ERROR_UNINITIALIZED: ZeusGPUInitError,
        pynvml.NVML_ERROR_INVALID_ARGUMENT: ZeusGPUInvalidArgError,
        pynvml.NVML_ERROR_NOT_SUPPORTED: ZeusGPUNotSupportedError,
        pynvml.NVML_ERROR_NO_PERMISSION: ZeusGPUNoPermissionError,
        pynvml.NVML_ERROR_ALREADY_INITIALIZED: ZeusGPUAlreadyInitializedError,
        pynvml.NVML_ERROR_NOT_FOUND: ZeusGPUNotFoundError,
        pynvml.NVML_ERROR_INSUFFICIENT_SIZE: ZeusGPUInsufficientSizeError,
        pynvml.NVML_ERROR_INSUFFICIENT_POWER: ZeusGPUInsufficientPowerError,
        pynvml.NVML_ERROR_DRIVER_NOT_LOADED: ZeusGPUDriverError,  # change to : ZeusGPUDriverNotLoadedError
        pynvml.NVML_ERROR_TIMEOUT: ZeusGPUTimeoutError,
        pynvml.NVML_ERROR_IRQ_ISSUE: ZeusGPUIRQError,
        pynvml.NVML_ERROR_LIBRARY_NOT_FOUND: ZeusGPULibraryNotFoundError,
        pynvml.NVML_ERROR_FUNCTION_NOT_FOUND: ZeusGPUFunctionNotFoundError,
        pynvml.NVML_ERROR_CORRUPTED_INFOROM: ZeusGPUCorruptedInfoROMError,
        pynvml.NVML_ERROR_GPU_IS_LOST: ZeusGPULostError,
        pynvml.NVML_ERROR_RESET_REQUIRED: ZeusGPUResetRequiredError,
        pynvml.NVML_ERROR_OPERATING_SYSTEM: ZeusGPUOperatingSystemError,
        pynvml.NVML_ERROR_LIB_RM_VERSION_MISMATCH: ZeusGPULibRMVersionMismatchError,
        pynvml.NVML_ERROR_MEMORY: ZeusGPUMemoryError,
        pynvml.NVML_ERROR_UNKNOWN: ZeusGPUUnknownError,
    }

    @_handle_nvml_errors
    def _get_handle(self):
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

    @_handle_nvml_errors
    def getPowerManagementLimitConstraints(self) -> tuple[int, int]:
        """Returns the minimum and maximum power management limits for the specified GPU. Units: mW."""
        return pynvml.nvmlDeviceGetPowerManagementLimitConstraints(self.handle)

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
    def setPowerManagementLimit(self, value: int | None = None) -> None:
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
    def setGpuLockedClocks(
        self, index: int, minMemClockMHz: int, maxMemClockMHz: int
    ) -> None:
        """Locks the GPU clock of the specified GPU to a range defined by the minimum and maximum GPU clock frequencies. Units: MHz."""
        pynvml.nvmlDeviceSetGpuLockedClocks(self.handle, minMemClockMHz, maxMemClockMHz)

    @_handle_nvml_errors
    def resetMemoryLockedClocks(self) -> None:
        """Resets the memory locked clocks of the specified GPU to their default values."""
        pynvml.nvmlDeviceResetMemoryLockedClocks(self.handle)

    @_handle_nvml_errors
    def resetGpuLockedClocks(self) -> None:
        """Resets the GPU locked clocks of the specified GPU to their default values."""
        pynvml.nvmlDeviceResetGpuLockedClocks(self.handle)

    @_handle_nvml_errors
    def getPowerUsage(self) -> int:
        """Returns the power usage of the specified GPU. Units: mW."""
        return pynvml.nvmlDeviceGetPowerUsage(self.handle)

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


def _handle_amdsmi_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except amdsmi.AmdSmiException as e:
            exception_class = AMDGPU._exception_map.get(e.value, ZeusGPUUnknownError)
            raise exception_class(e.msg) from e

    return wrapper


class AMDGPU(GPU):
    """Control a Single AMD GPU.

    Uses amdsmi Library to control and query GPU. There is a 1:1 mapping between the methods in this class and the amdsmi library functions.
    Zeus GPU Exceptions are raised when amdsmi errors occur.
    To ensure computational efficiency, this class utilizes caching (ex. saves the handle) to avoid repeated calls to amdsmi.
    """

    def __init__(self, gpu_index: int) -> None:
        """Initializes the AMDGPU object with a specified GPU index. Acquires a handle to the GPU using `amdsmi.amdsmi_get_processor_handles()`."""
        super().__init__(gpu_index)
        self._get_handle()

    _exception_map = {}

    @_handle_amdsmi_errors
    def _get_handle(self):
        self.handle = amdsmi.amdsmi_get_processor_handles()[self.gpu_index]

    @_handle_amdsmi_errors
    def getPowerManagementLimitConstraints(self) -> tuple[int, int]:
        """Returns the minimum and maximum power management limits for the specified GPU. Units: mW."""
        info = amdsmi.amdsmi_get_power_cap_info(self.handle)
        return (info.min_power_cap, info.max_power_cap)

    @_handle_amdsmi_errors
    def setPersistenceMode(self, enable: bool) -> None:
        """Enables persistence mode for the specified GPU."""
        raise ZeusGPUNotSupportedError(
            "Persistence mode is not supported for AMD GPUs yet"
        )
        profile = ...  # TODO: find out correct profile
        amdsmi.amdsmi_set_gpu_power_profile(self.handle, 0, profile)

    @_handle_amdsmi_errors
    def setPowerManagementLimit(self, value: int) -> None:
        """Sets the power management limit for the specified GPU to the given value. Unit: mW."""
        amdsmi.amdsmi_set_power_cap(self.handle, sensor_id=0, cap=value)

    @_handle_amdsmi_errors
    def resetPowerManagementLimit(self) -> None:
        """Resets the power management limit for the specified GPU to the default value."""
        info = amdsmi.amdsmi_get_power_cap_info(self.handle)
        amdsmi.amdsmi_set_power_cap(
            self.handle, sensor_id=0, cap=info.default_power_cap
        )

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
        num_supported, current, frequency = amdsmi.amdsmi_get_clk_freq(
            self.handle, clk_type=amdsmi.AmdSmiClkType.MEM
        )  # TODO: Figure out correct clk_type
        # frequency; List of frequencies, only the first num_supported frequencies are valid"""
        return frequency[:num_supported]

    @_handle_amdsmi_errors
    def getSupportedGraphicsClocks(self, freq: int) -> list[int]:
        """Returns a list of supported graphics clock frequencies for the specified GPU at a given frequency. Units: MHz."""
        raise ZeusGPUNotSupportedError(
            "Getting supported graphics clocks is not supported for AMD GPUs yet"
        )

    @_handle_amdsmi_errors
    def getName(self) -> str:
        """Returns the name of the specified GPU."""
        (
            market_name,
            vendor_id,
            device_id,
            rev_id,
            asic_serial,
        ) = amdsmi.amdsmi_get_gpu_asic_info(
            self.handle
        )  # TODO: Does this return correct string
        return market_name

    @_handle_amdsmi_errors
    def setGpuLockedClocks(
        self, index: int, minMemClockMHz: int, maxMemClockMHz: int
    ) -> None:
        """Locks the GPU clock of the specified GPU to a range defined by the minimum and maximum GPU clock frequencies.  Units: MHz."""
        amdsmi.amdsmi_set_gpu_clk_range(
            self.handle,
            minMemClockMHz,
            maxMemClockMHz,
            clk_type=amdsmi.AMDSMI_CLK_TYPE_GFX,
        )

    @_handle_amdsmi_errors
    def resetMemoryLockedClocks(self) -> None:
        """Resets the memory locked clocks of the specified GPU to their default values."""
        amdsmi.amdsmi_reset_gpu_clk(
            self.handle, clk_type=amdsmi.AMDSMI_CLK_TYPE_SYS
        )  # TODO: check docs

    @_handle_amdsmi_errors
    def resetGpuLockedClocks(self) -> None:
        """Resets the GPU locked clocks of the specified GPU to their default values."""
        amdsmi.amdsmi_reset_gpu_clk(
            self.handle, clk_type=amdsmi.AMDSMI_CLK_TYPE_GFX
        )  # TODO: check docs

    @_handle_amdsmi_errors
    def getPowerUsage(self) -> int:
        """Returns the power usage of the specified GPU. Units: mW."""
        raise ZeusGPUNotSupportedError(
            "Getting power usage is not supported for AMD GPUs yet"
        )

    @_handle_amdsmi_errors
    def supportsGetTotalEnergyConsumption(self) -> bool:
        """Returns True if the specified GPU supports retrieving the total energy consumption."""
        raise ZeusGPUNotSupportedError(
            "Getting total energy consumption is not supported for AMD GPUs yet"
        )

    @_handle_amdsmi_errors
    def getTotalEnergyConsumption(self) -> int:
        """Returns the total energy consumption of the specified GPU. Units: mJ."""
        raise ZeusGPUNotSupportedError(
            "Getting total energy consumption is not supported for AMD GPUs yet"
        )


class UnprivilegedAMDGPU(AMDGPU):
    """Control a Single AMD GPU with no SYS_ADMIN privileges.

    Uses amdsmi Library to control and query GPU. There is a 1:1 mapping between the methods in this class and the amdsmi library functions.
    Zeus GPU Exceptions are raised when amdsmi errors occur.
    To ensure computational efficiency, this class utilizes caching (ex. saves the handle) to avoid repeated calls to amdsmi.
    """

    pass


class GPUs(abc.ABC):
    """An abstract base class for GPU manager object.

    This class defines the essential interface and common functionality for GPU management, instantiating multiple `GPU` objects for each GPU being tracked.
    Forwards the call for a specific method to the corresponding GPU object.
    """

    @abc.abstractmethod
    def __init__(self, ensure_homogeneous: bool = False) -> None:
        """Initializes the GPU management library to communicate with the GPU driver and sets up tracking for specified GPUs."""
        pass

    @abc.abstractmethod
    def __del__(self) -> None:
        """Shuts down the GPU monitoring library to release resources and clean up."""
        pass

    @property
    @abc.abstractmethod
    def gpus(self) -> list[GPU]:
        """Returns a list of GPU objects being tracked."""
        pass

    def _ensure_homogeneous(self) -> None:
        """Ensures that all tracked GPUs are homogeneous in terms of name."""
        gpu_names = [gpu.getName() for gpu in self.gpus]
        # Both zero (no GPUs found) and one are fine.
        if len(set(gpu_names)) > 1:
            raise ZeusBaseGPUError(f"Heterogeneous GPUs found: {gpu_names}")

    def getPowerManagementLimitConstraints(self, index: int) -> tuple[int, int]:
        """Returns the minimum and maximum power management limits for the specified GPU. Units: mW."""
        return self.gpus[index].getPowerManagementLimitConstraints()

    def setPersistenceMode(self, index: int, enable: bool) -> None:
        """Enables persistence mode for the specified GPU."""
        self.gpus[index].setPersistenceMode(enable)

    def setPowerManagementLimit(self, index: int, value: int) -> None:
        """Sets the power management limit for the specified GPU to the given value. Unit: mW."""
        self.gpus[index].setPowerManagementLimit(value)

    def resetPowerManagementLimit(self, index: int) -> None:
        """Resets the power management limit for the specified GPU to the default value."""
        self.gpus[index].resetPowerManagementLimit()

    def setMemoryLockedClocks(
        self, index: int, minMemClockMHz: int, maxMemClockMHz: int
    ) -> None:
        """Locks the memory clock of the specified GPU to a range defined by the minimum and maximum memory clock frequencies. Units: MHz."""
        self.gpus[index].setMemoryLockedClocks(minMemClockMHz, maxMemClockMHz)

    def getSupportedMemoryClocks(self, index: int) -> list[int]:
        """Returns a list of supported memory clock frequencies for the specified GPU. Units: MHz."""
        self.gpus[index].getSupportedMemoryClocks()

    def getSupportedGraphicsClocks(self, index: int, freq: int) -> list[int]:
        """Returns a list of supported graphics clock frequencies for the specified GPU at a given frequency. Units: MHz."""
        self.gpus[index].getSupportedGraphicsClocks(freq)

    def getName(self, index: int) -> str:
        """Returns the name of the specified GPU."""
        return self.gpus[index].getName()

    def setGpuLockedClocks(
        self, index: int, minMemClockMHz: int, maxMemClockMHz: int
    ) -> None:
        """Locks the GPU clock of the specified GPU to a range defined by the minimum and maximum GPU clock frequencies. Units: MHz."""
        self.gpus[index].setGpuLockedClocks(minMemClockMHz, maxMemClockMHz)

    def resetMemoryLockedClocks(self, index: int) -> None:
        """Resets the memory locked clocks of the specified GPU to their default values."""
        self.gpus[index].resetMemoryLockedClocks()

    def resetGpuLockedClocks(self, index: int) -> None:
        """Resets the GPU locked clocks of the specified GPU to their default values."""
        self.gpus[index].resetGpuLockedClocks()

    def getPowerUsage(self, index: int) -> int:
        """Returns the power usage of the specified GPU. Units: mW."""
        return self.gpus[index].getPowerUsage()

    def supportsGetTotalEnergyConsumption(self, index: int) -> bool:
        """Returns True if the specified GPU supports retrieving the total energy consumption."""
        return self.gpus[index].supportsGetTotalEnergyConsumption()

    def getTotalEnergyConsumption(self, index: int) -> int:
        """Returns the total energy consumption of the specified GPU. Units: mJ."""
        return self.gpus[index].getTotalEnergyConsumption()

    def __len__(self) -> int:
        """Returns the number of GPUs being tracked."""
        return len(self.gpus)


class NVIDIAGPUs(GPUs):
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
            ensure_homogeneous (bool, optional): If True, ensures that all tracked GPUs have the same name (return value of pynvml.nvmlDeviceGetName). False by default.
        """
        try:
            pynvml.nvmlInit()
            self._init_gpus()
            if ensure_homogeneous:
                self._ensure_homogeneous()
        except pynvml.NVMLError as e:
            exception_class = NVIDIAGPU._exception_map.get(e.value, ZeusBaseGPUError)
            raise exception_class(e.msg) from e

    @property
    def gpus(self) -> list[GPU]:
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

        # eventually replace with: self.gpus = [NVIDIAGPU(gpu_num) for gpu_num in self.visible_indices]

    def __del__(self) -> None:
        """Shuts down the NVIDIA GPU monitoring library to release resources and clean up."""
        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlShutdown()


class AMDGPUs(GPUs):
    """AMD GPU Manager object, containing individual AMDGPU objects, abstracting amdsmi calls and handling related exceptions.

    This class provides a high-level interface to interact with AMD GPUs. `ROCR_VISIBLE_DEVICES` environment variable is respected if set. For example, if there are
    4 GPUs and `ROCR_VISIBLE_DEVICES=0,2`, only GPUs 0 and 2 are instantiated. In this case, to access
    GPU of ROCR index 0, use the index 0, and for ROCR index 2, use the index 1.

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

    Note: This class instantiates (grabs the handle, by calling `amdsmi.amdsmi_get_processor_handles()`) all GPUs that are visible to the system, as determined by the `ROCR_VISIBLE_DEVICES` environment variable if set.

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
            exception_class = AMDGPU._exception_map.get(e.value, ZeusBaseGPUError)
            raise exception_class(e.msg) from e

    def _init_gpus(self) -> None:
        # Must respect `ROCR_VISIBLE_DEVICES` if set
        if (visible_device := os.environ.get("ROCR_VISIBLE_DEVICES")) is not None:
            self.visible_indices = [int(idx) for idx in visible_device.split(",")]
        else:
            self.visible_indices = list(
                range(len(amdsmi.amdsmi_get_processor_handles()))
            )

    def __del__(self) -> None:
        """Shuts down the AMD GPU monitoring library to release resources and clean up."""
        with contextlib.suppress(amdsmi.AmdSmiException):
            amdsmi.amdsmi_shut_down()  # Ignore error on shutdown. Neccessary for proper cleanup and test functionality


_gpus: GPUs | None = None


def get_gpus(ensure_homogeneous: bool = False) -> GPUs:
    """Initialize and return a singleton GPU monitoring object for NVIDIA or AMD GPUs.

    The function returns a GPU management object that aims to abstract the underlying GPU monitoring libraries
    (pynvml for NVIDIA GPUs and amdsmi for AMD GPUs), and provides a 1:1 mapping between the methods in the object and related library functions.

    This function attempts to initialize GPU monitoring using the pynvml library for NVIDIA GPUs
    first. If pynvml is not available or fails to initialize, it then tries to use the amdsmi
    library for AMD GPUs. If both attempts fail, it raises a ZeusErrorInit exception.

    Args:
        ensure_homogeneous (bool, optional): If True, ensures that all tracked GPUs have the same name. False by default.
    """
    global _gpus
    if _gpus is not None:
        return _gpus

    if pynvml_is_available():
        _gpus = NVIDIAGPUs(ensure_homogeneous)
        return _gpus
    elif amdsmi_is_available():
        _gpus = AMDGPUs(ensure_homogeneous)
        return _gpus
    else:
        raise ZeusGPUInitError(
            "Failed to initialize GPU monitoring for NVIDIA and AMD GPUs."
        )
