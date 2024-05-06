"""GPU management module for Zeus."""

from __future__ import annotations
import abc
import functools
import os
import contextlib
from typing import Sequence

import pynvml

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

from zeus.device.exception import ZeusBaseGPUError
from zeus.utils.logging import get_logger

logger = get_logger(name=__name__)


# EXCEPTION WRAPPERS


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


class ZeusGPUDriverNotLoadedError(ZeusBaseGPUError):
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


# SINGLE GPU MANAGEMENT OBJECTS


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
    def setPowerManagementLimit(self, value: int) -> None:
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
    def setGpuLockedClocks(self, minGpuClockMHz: int, maxGpuClockMHz: int) -> None:
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
    def getInstantPowerUsage(self) -> int:
        """Returns the current power usage of the GPU. Units: mW."""
        pass

    @abc.abstractmethod
    def supportsGetTotalEnergyConsumption(self) -> bool:
        """Check if the GPU supports retrieving total energy consumption."""
        pass

    @abc.abstractmethod
    def getTotalEnergyConsumption(self) -> int:
        """Return the total energy consumption of the GPU since driver load. Units: mJ."""
        pass


# GPU MANAGEMENT OBJECTS


def _handle_nvml_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except pynvml.NVMLError as e:
            exception_class = NVIDIAGPU._exception_map.get(
                e.value,  # pyright: ignore[reportAttributeAccessIssue]
                ZeusGPUUnknownError,
            )
            raise exception_class(str(e)) from e

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
        pynvml.NVML_ERROR_DRIVER_NOT_LOADED: ZeusGPUDriverNotLoadedError,
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


def _handle_amdsmi_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except amdsmi.AmdSmiLibraryException as e:
            exception_class = AMDGPU._exception_map.get(
                e.get_error_code(), ZeusGPUUnknownError
            )
            raise exception_class(e.get_error_info()) from e

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
        self._supportsGetTotalEnergyConsumption = None

    _exception_map = {
        1: ZeusGPUInvalidArgError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL
        2: ZeusGPUNotSupportedError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED
        8: ZeusGPUTimeoutError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_TIMEOUT
        10: ZeusGPUNoPermissionError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM
        15: ZeusGPUMemoryError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_OUT_OF_RESOURCES
        18: ZeusGPUInitError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INIT_ERROR
        31: ZeusGPUNotFoundError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_FOUND
        32: ZeusGPUInitError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_INIT
        34: ZeusGPUDriverNotLoadedError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_DRIVER_NOT_LOADED
        41: ZeusGPUInsufficientSizeError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INSUFFICIENT_SIZE
        45: ZeusGPUDriverNotLoadedError,  # amdsmi.amdsmi_wrapper.AMDSMI_NO_ENERGY_DRV
        46: ZeusGPUDriverNotLoadedError,  # amdsmi.amdsmi_wrapper.AMDSMI_NO_MSR_DRV
        47: ZeusGPUDriverNotLoadedError,  # amdsmi.amdsmi_wrapper.AMDSMI_NO_HSMP_DRV
        48: ZeusGPUNotSupportedError,  # amdsmi.amdsmi_wrapper.AMDSMI_NO_HSMP_SUP
        49: ZeusGPUNotSupportedError,  # amdsmi.amdsmi_wrapper.AMDSMI_NO_HSMP_MSG_SUP
        50: ZeusGPUTimeoutError,  # amdsmi.amdsmi_wrapper.AMDSMI_HSMP_TIMEOUT
        51: ZeusGPUDriverNotLoadedError,  # amdsmi.amdsmi_wrapper.AMDSMI_NO_DRV
        52: ZeusGPULibraryNotFoundError,  # amdsmi.amdsmi_wrapper.AMDSMI_FILE_NOT_FOUND
        53: ZeusGPUInvalidArgError,  # amdsmi.amdsmi_wrapper.AMDSMI_ARG_PTR_NULL
        4294967295: ZeusGPUUnknownError,  # amdsmi.amdsmi_wrapper.AMDSMI_STATUS_UNKNOWN_ERROR
    }

    @_handle_amdsmi_errors
    def _get_handle(self):
        handles = amdsmi.amdsmi_get_processor_handles()
        if len(handles) <= self.gpu_index:
            raise ZeusGPUNotFoundError(
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
        raise ZeusGPUNotSupportedError(
            "AMDSMI does not support querying memory frequencies"
        )

    @_handle_amdsmi_errors
    def getSupportedGraphicsClocks(self, freq: int) -> list[int]:
        """Returns a list of supported graphics clock frequencies for the specified GPU at a given frequency. Units: MHz."""
        raise ZeusGPUNotSupportedError(
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
    def gpus(self) -> Sequence[GPU]:
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
        return self.gpus[index].getSupportedMemoryClocks()

    def getSupportedGraphicsClocks(self, index: int, freq: int) -> list[int]:
        """Returns a list of supported graphics clock frequencies for the specified GPU at a given frequency. Units: MHz."""
        return self.gpus[index].getSupportedGraphicsClocks(freq)

    def getName(self, index: int) -> str:
        """Returns the name of the specified GPU."""
        return self.gpus[index].getName()

    def setGpuLockedClocks(
        self, index: int, minGpuClockMHz: int, maxGpuClockMHz: int
    ) -> None:
        """Locks the GPU clock of the specified GPU to a range defined by the minimum and maximum GPU clock frequencies. Units: MHz."""
        self.gpus[index].setGpuLockedClocks(minGpuClockMHz, maxGpuClockMHz)

    def resetMemoryLockedClocks(self, index: int) -> None:
        """Resets the memory locked clocks of the specified GPU to their default values."""
        self.gpus[index].resetMemoryLockedClocks()

    def resetGpuLockedClocks(self, index: int) -> None:
        """Resets the GPU locked clocks of the specified GPU to their default values."""
        self.gpus[index].resetGpuLockedClocks()

    def getInstantPowerUsage(self, index: int) -> int:
        """Returns the power usage of the specified GPU. Units: mW."""
        return self.gpus[index].getInstantPowerUsage()

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
            ensure_homogeneous (bool): If True, ensures that all tracked GPUs have the same name (return value of `nvmlDeviceGetName`). False by default.
        """
        try:
            pynvml.nvmlInit()
            self._init_gpus()
            if ensure_homogeneous:
                self._ensure_homogeneous()
        except pynvml.NVMLError as e:
            exception_class = NVIDIAGPU._exception_map.get(
                e.value, ZeusBaseGPUError  # pyright: ignore[reportAttributeAccessIssue]
            )
            raise exception_class(
                e.msg  # pyright: ignore[reportAttributeAccessIssue]
            ) from e

    @property
    def gpus(self) -> Sequence[GPU]:
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


class AMDGPUs(GPUs):
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
            exception_class = AMDGPU._exception_map.get(e.value, ZeusBaseGPUError)
            raise exception_class(e.msg) from e

    @property
    def gpus(self) -> Sequence[GPU]:
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

    if nvml_is_available():
        _gpus = NVIDIAGPUs(ensure_homogeneous)
        return _gpus
    elif amdsmi_is_available():
        _gpus = AMDGPUs(ensure_homogeneous)
        return _gpus
    else:
        raise ZeusGPUInitError(
            "NVML and AMDSMI unavailable. Failed to initialize GPU management library."
        )


def nvml_is_available() -> bool:
    """Check if PyNVML is available."""
    try:
        import pynvml
    except ImportError:
        logger.info("PyNVML is not available.")
        return False
    try:
        pynvml.nvmlInit()
        logger.info("PyNVML is available and initialized.")
        return True
    except pynvml.NVMLError:
        logger.info("PyNVML is available but could not initialize.")
        return False


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
