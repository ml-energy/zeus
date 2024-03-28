"""GPU management module for Zeus.

Class hierarchy
The abstraction provided by this module (most importantly, get_gpus). Basically, how a Zeus-developer should use this module.
How different GPU vendors are handled



"""

from __future__ import annotations
import abc
import functools
import os
from typing import TYPE_CHECKING
import contextlib

import pynvml # necessary for testing to mock! 

from zeus.device.exception import ZeusBaseGPUError
from zeus.util import pynvml_is_available, amdsmi_is_available

if TYPE_CHECKING:
    import amdsmi

""" EXCEPTION WRAPPERS """


class ZeusInitGPUError(ZeusBaseGPUError):
    """Import error or GPU library initialization failures."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusInvalidArgGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Invalid Argument."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusNotSupportedGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Not Supported Operation on GPU."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusNoPermissionGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for No Permission to perform GPU operation."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusAlreadyInitializedGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Already Initialized GPU."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusNotFoundGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Not Found GPU."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusInsufficientSizeGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Insufficient Size of GPU."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusInsufficientPowerGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Insufficient Power of GPU."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusDriverErrorGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Driver Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusTimeoutGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Timeout Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusIRQErrorGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for IRQ Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusLibraryNotFoundGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Library Not Found Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusFunctionNotFoundGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Function Not Found Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusCorruptedInfoROMGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Corrupted Info ROM Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusLostGPUGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Lost GPU Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusResetRequiredGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Reset Required Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusOperatingSystemGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Operating System Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusLibRMVersionMismatchGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for LibRM Version Mismatch Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusMemoryGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Insufficient Memory Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusUnknownGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Unknown Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


""" SINGLE GPU MANAGEMENT OBJECTS """


class GPU(abc.ABC):
    # TODO: Fix doctring
    """Abstract base class for GPU management. This class defines the interface for interacting with GPUs, including power management, clock settings, and information retrieval. Subclasses should implement the methods to interact with specific GPU libraries (e.g., NVML for NVIDIA GPUs)."""

    def __init__(self, gpu_index: int) -> None:
        """Initialize the GPU with a specified index."""
        self.gpu_index = gpu_index

    @abc.abstractmethod
    def getPowerManagementLimitConstraints(self) -> tuple[int, int]:
        """Return the minimum and maximum power management limits for the GPU."""
        pass

    @abc.abstractmethod
    def setPersistenceMode(self, enable: bool) -> None:
        """Enable persistence mode for the GPU."""
        pass

    @abc.abstractmethod
    def setPowerManagementLimit(self, value: int = None) -> None:
        """Set the power management limit for the GPU to a specified value or default."""
        pass

    @abc.abstractmethod
    def setMemoryLockedClocks(self, minMemClockMHz: int, maxMemClockMHz: int) -> None:
        """Lock the memory clock to a specified range."""
        pass

    @abc.abstractmethod
    def getSupportedMemoryClocks(self) -> list[int]:
        """Return a list of supported memory clock frequencies for the GPU."""
        pass

    @abc.abstractmethod
    def getSupportedGraphicsClocks(self, freq: int) -> list[int]:
        """Return a list of supported graphics clock frequencies for a given memory frequency."""
        pass

    @abc.abstractmethod
    def getName(self) -> str:
        """Return the name of the GPU."""
        pass

    @abc.abstractmethod
    def setGpuLockedClocks(
        self, index: int, minMemClockMHz: int, maxMemClockMHz: int
    ) -> None:
        """Lock the GPU clock to a specified range."""
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
        """Return the current power usage of the GPU."""
        pass

    @abc.abstractmethod
    def supportsGetTotalEnergyConsumption(self) -> bool:
        """Check if the GPU supports retrieving total energy consumption."""
        pass

    @abc.abstractmethod
    def getTotalEnergyConsumption(self) -> int:
        """Return the total energy consumption of the GPU since driver load."""
        pass


""" GPU MANAGEMENT OBJECTS """


class NVIDIAGPU(GPU):
    """A subclass of GPU tailored for NVIDIA GPUs, providing an interface to control and query GPU features via the NVML library.

    This class includes methods to set and get power management limits, persistence mode, memory and GPU clock speeds,
    and to query supported memory and graphics clock speeds, GPU names, and power usage. Exception handling for NVML
    errors is integrated, mapping NVML error codes to custom exceptions for clearer error reporting and management.
    """

    _exception_map = {
        pynvml.NVML_ERROR_UNINITIALIZED: ZeusInitGPUError,
        pynvml.NVML_ERROR_INVALID_ARGUMENT: ZeusInvalidArgGPUError,
        pynvml.NVML_ERROR_NOT_SUPPORTED: ZeusNotSupportedGPUError,
        pynvml.NVML_ERROR_NO_PERMISSION: ZeusNoPermissionGPUError,
        pynvml.NVML_ERROR_ALREADY_INITIALIZED: ZeusAlreadyInitializedGPUError,
        pynvml.NVML_ERROR_NOT_FOUND: ZeusNotFoundGPUError,
        pynvml.NVML_ERROR_INSUFFICIENT_SIZE: ZeusInsufficientSizeGPUError,
        pynvml.NVML_ERROR_INSUFFICIENT_POWER: ZeusInsufficientPowerGPUError,
        pynvml.NVML_ERROR_DRIVER_NOT_LOADED: ZeusDriverErrorGPUError, # change to : ZeusGPUDriverNotLoadedError
        pynvml.NVML_ERROR_TIMEOUT: ZeusTimeoutGPUError,
        pynvml.NVML_ERROR_IRQ_ISSUE: ZeusIRQErrorGPUError,
        pynvml.NVML_ERROR_LIBRARY_NOT_FOUND: ZeusLibraryNotFoundGPUError,
        pynvml.NVML_ERROR_FUNCTION_NOT_FOUND: ZeusFunctionNotFoundGPUError,
        pynvml.NVML_ERROR_CORRUPTED_INFOROM: ZeusCorruptedInfoROMGPUError,
        pynvml.NVML_ERROR_GPU_IS_LOST: ZeusLostGPUGPUError,
        pynvml.NVML_ERROR_RESET_REQUIRED: ZeusResetRequiredGPUError,
        pynvml.NVML_ERROR_OPERATING_SYSTEM: ZeusOperatingSystemGPUError,
        pynvml.NVML_ERROR_LIB_RM_VERSION_MISMATCH: ZeusLibRMVersionMismatchGPUError,
        pynvml.NVML_ERROR_MEMORY: ZeusMemoryGPUError,
        pynvml.NVML_ERROR_UNKNOWN: ZeusUnknownGPUError,
    }

    @staticmethod
    def _handle_nvml_errors(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except pynvml.NVMLError as e:
                exception_class = NVIDIAGPU._exception_map.get(
                    e.value, ZeusBaseGPUError
                )
                raise exception_class(e.msg) from e

        return wrapper

    @_handle_nvml_errors
    def _get_handle(self):
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

    def __init__(self, gpu_index: int) -> None:
        """Initializes the NVIDIAGPU object with a specified GPU index. Acquires a handle to the GPU using `pynvml.nvmlDeviceGetHandleByIndex`."""
        super().__init__(gpu_index)
        self._get_handle()

    @_handle_nvml_errors
    def getPowerManagementLimitConstraints(self) -> tuple[int, int]:
        """Returns the minimum and maximum power management limits for the specified GPU."""
        return pynvml.nvmlDeviceGetPowerManagementLimitConstraints(self.handle)

    @_handle_nvml_errors
    def setPersistenceMode(self, enable: bool) -> None:
        """If enable = True, enables persistence mode for the specified GPU. If enable = False, disables persistence mode."""
        if enable:
            pynvml.nvmlDeviceSetPersistenceMode(self.handle, pynvml.NVML_FEATURE_ENABLED)
        else:
            pynvml.nvmlDeviceSetPersistenceMode(self.handle, pynvml.NVML_FEATURE_DISABLED)

    @_handle_nvml_errors
    def setPowerManagementLimit(self, value: int | None = None) -> None:
        """Sets the power management limit for the specified GPU to the given value."""
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
        """Locks the memory clock of the specified GPU to a range defined by the minimum and maximum memory clock frequencies."""
        pynvml.nvmlDeviceSetMemoryLockedClocks(
            self.handle, minMemClockMHz, maxMemClockMHz
        )

    @_handle_nvml_errors
    def getSupportedMemoryClocks(self) -> list[int]:
        """Returns a list of supported memory clock frequencies for the specified GPU."""
        return pynvml.nvmlDeviceGetSupportedMemoryClocks(self.handle)

    @_handle_nvml_errors
    def getSupportedGraphicsClocks(self, freq: int) -> list[int]:
        """Returns a list of supported graphics clock frequencies for the specified GPU at a given frequency."""
        return pynvml.nvmlDeviceGetSupportedGraphicsClocks(self.handle, freq)

    @_handle_nvml_errors
    def getName(self) -> str:
        """Returns the name of the specified GPU."""
        return pynvml.nvmlDeviceGetName(self.handle)

    @_handle_nvml_errors
    def setGpuLockedClocks(
        self, index: int, minMemClockMHz: int, maxMemClockMHz: int
    ) -> None:
        """Locks the GPU clock of the specified GPU to a range defined by the minimum and maximum GPU clock frequencies."""
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
        """Returns the power usage of the specified GPU."""
        return pynvml.nvmlDeviceGetPowerUsage(self.handle)

    @_handle_nvml_errors
    def supportsGetTotalEnergyConsumption(self) -> bool:
        """Returns True if the specified GPU supports retrieving the total energy consumption."""
        # NVIDIA GPUs Volta or newer support this method
        return (
            pynvml.nvmlDeviceGetArchitecture(self.handle)
            >= pynvml.NVML_DEVICE_ARCH_VOLTA
        )

    @_handle_nvml_errors
    def getTotalEnergyConsumption(self) -> int:
        """Returns the total energy consumption of the specified GPU."""
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)


class UnprivilegedNVIDIAGPU(NVIDIAGPU):
    """A subclass of NVIDIAGPU that does not require root privileges."""

    def __init__(self, gpu_index: int) -> None:
        """Initializes the UnprivilegedNVIDIAGPU object with a specified GPU index. Acquires a handle to the GPU using `pynvml.nvmlDeviceGetHandleByIndex`."""
        super().__init__(gpu_index)


class AMDGPU(GPU):
    """A subclass of GPU tailored for AMD GPUs, providing an interface to control and query GPU features via the amdsmi library.

    This class includes methods to set and get power management limits, persistence mode, memory and GPU clock speeds,
    and to query supported memory and graphics clock speeds, GPU names, and power usage. Exception handling for amdsmi
    errors is integrated, mapping amdsmi error codes to custom exceptions for clearer error reporting and management.
    """

    _exception_map = {}

    @staticmethod
    def _handle_amdsmi_errors(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except amdsmi.AmdSmiException as e:
                exception_class = AMDGPU._exception_map.get(e.value, ZeusBaseGPUError)
                raise exception_class(e.msg) from e

        return wrapper

    def __init__(self, gpu_index: int) -> None:
        """Initializes the AMDGPU object with a specified GPU index. Acquires a handle to the GPU using `amdsmi.amdsmi_get_processor_handles()`."""
        super().__init__(gpu_index)
        self._get_handle()

    @_handle_amdsmi_errors
    def _get_handle(self):
        self.handle = amdsmi.amdsmi_get_processor_handles()[self.gpu_index]

    @_handle_amdsmi_errors
    def getPowerManagementLimitConstraints(self) -> tuple[int, int]:
        """Returns the minimum and maximum power management limits for the specified GPU."""
        info = amdsmi.amdsmi_get_power_cap_info(self.handle)
        return (info.min_power_cap, info.max_power_cap)

    @_handle_amdsmi_errors
    def setPersistenceMode(self, enable: bool) -> None:
        """Enables persistence mode for the specified GPU."""
        raise ZeusNotSupportedGPUError(
            "Persistence mode is not supported for AMD GPUs yet"
        )
        profile = ...  # TODO: find out correct profile
        amdsmi.amdsmi_set_gpu_power_profile(self.handle, 0, profile)

    @_handle_amdsmi_errors
    def setPowerManagementLimit(self, value: int) -> None:
        """Sets the power management limit for the specified GPU to the given value."""
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
        """Locks the memory clock of the specified GPU to a range defined by the minimum and maximum memory clock frequencies."""
        amdsmi.amdsmi_set_gpu_clk_range(
            self.handle,
            minMemClockMHz,
            maxMemClockMHz,
            clk_type=amdsmi.AmdSmiClkType.MEM,
        )

    @_handle_amdsmi_errors
    def getSupportedMemoryClocks(self) -> list[int]:
        """Returns a list of supported memory clock frequencies for the specified GPU."""
        num_supported, current, frequency = amdsmi.amdsmi_get_clk_freq(
            self.handle, clk_type=amdsmi.AmdSmiClkType.MEM
        )  # TODO: Figure out correct clk_type
        # frequency; List of frequencies, only the first num_supported frequencies are valid"""
        return frequency[:num_supported]

    @_handle_amdsmi_errors
    def getSupportedGraphicsClocks(self, freq: int) -> list[int]:
        """Returns a list of supported graphics clock frequencies for the specified GPU at a given frequency."""
        raise ZeusNotSupportedGPUError(
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
        """Locks the GPU clock of the specified GPU to a range defined by the minimum and maximum GPU clock frequencies."""
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
        """Returns the power usage of the specified GPU."""
        raise ZeusNotSupportedGPUError(
            "Getting power usage is not supported for AMD GPUs yet"
        )

    @_handle_amdsmi_errors
    def supportsGetTotalEnergyConsumption(self) -> bool:
        """Returns True if the specified GPU supports retrieving the total energy consumption."""
        raise ZeusNotSupportedGPUError(
            "Getting total energy consumption is not supported for AMD GPUs yet"
        )

    @_handle_amdsmi_errors
    def getTotalEnergyConsumption(self) -> int:
        """Returns the total energy consumption of the specified GPU."""
        raise ZeusNotSupportedGPUError(
            "Getting total energy consumption is not supported for AMD GPUs yet"
        )


class UnprivilegedAMDGPU(AMDGPU):
    """A subclass of AMDGPU that does not require root privileges."""

    pass


class GPUs(abc.ABC):
    """An abstract base class for managing and interacting with GPUs.

    This class defines the essential interface and common functionality for GPU management, including power management, clock settings, and information retrieval.
    """

    @abc.abstractmethod
    def __init__(self) -> None:
        """Initializes the GPU management library to communicate with the GPU driver and sets up tracking for specified GPUs."""
        pass

    @abc.abstractmethod
    def __del__(self) -> None:
        """Shuts down the GPU monitoring library to release resources and clean up."""
        pass

    # @abc.abstractproperty
    # gpus = None
    # def gpus(self) -> dict[int, GPU]:
    #     """Returns a dictionary of GPU objects being tracked."""
    #     pass

    def _ensure_homogeneous(self) -> None:
        """Ensures that all tracked GPUs are homogeneous in terms of name."""
        
        gpu_names = [gpu.getName() for gpu in self.gpus.values()]
        # Both zero (no GPUs found) and one are fine.
        if len(set(gpu_names)) > 1:
            raise ZeusBaseGPUError(f"Heterogeneous GPUs found: {gpu_names}")

    def getPowerManagementLimitConstraints(self, index: int) -> tuple[int, int]:
        """Returns the minimum and maximum power management limits for the specified GPU."""
        return self.gpus[index].getPowerManagementLimitConstraints()

    def setPersistenceMode(self, index: int, enable: bool) -> None:
        """If enable = True, enables persistence mode for the specified GPU. If enable = False, disables persistence mode."""
        self.gpus[index].setPersistenceMode(enable)

    def setPowerManagementLimit(self, index: int, value: int) -> None:
        """Sets the power management limit for the specified GPU to the given value."""
        self.gpus[index].setPowerManagementLimit(value)
    
    def resetPowerManagementLimit(self, index: int) -> None:
        """Resets the power management limit for the specified GPU to the default value."""
        self.gpus[index].resetPowerManagementLimit()

    def setMemoryLockedClocks(
        self, index: int, minMemClockMHz: int, maxMemClockMHz: int
    ) -> None:
        """Locks the memory clock of the specified GPU to a range defined by the minimum and maximum memory clock frequencies."""
        self.gpus[index].setMemoryLockedClocks(minMemClockMHz, maxMemClockMHz)

    def getSupportedMemoryClocks(self, index: int) -> list[int]:
        """Returns a list of supported memory clock frequencies for the specified GPU."""
        self.gpus[index].getSupportedMemoryClocks()

    def getSupportedGraphicsClocks(self, index: int, freq: int) -> list[int]:
        """Returns a list of supported graphics clock frequencies for the specified GPU at a given frequency."""
        self.gpus[index].getSupportedGraphicsClocks(freq)

    def getName(self, index: int) -> str:
        """Returns the name of the specified GPU."""
        return self.gpus[index].getName()

    def setGpuLockedClocks(
        self, index: int, minMemClockMHz: int, maxMemClockMHz: int
    ) -> None:
        """Locks the GPU clock of the specified GPU to a range defined by the minimum and maximum GPU clock frequencies."""
        self.gpus[index].setGpuLockedClocks(minMemClockMHz, maxMemClockMHz)

    def resetMemoryLockedClocks(self, index: int) -> None:
        """Resets the memory locked clocks of the specified GPU to their default values."""
        self.gpus[index].resetMemoryLockedClocks()

    def resetGpuLockedClocks(self, index: int) -> None:
        """Resets the GPU locked clocks of the specified GPU to their default values."""
        self.gpus[index].resetGpuLockedClocks()

    def getPowerUsage(self, index: int) -> int:
        """Returns the power usage of the specified GPU."""
        return self.gpus[index].getPowerUsage()

    def supportsGetTotalEnergyConsumption(self, index: int) -> bool:
        """Returns True if the specified GPU supports retrieving the total energy consumption."""
        return self.gpus[index].supportsGetTotalEnergyConsumption()

    def getTotalEnergyConsumption(self, index: int) -> int:
        """Returns the total energy consumption of the specified GPU."""
        return self.gpus[index].getTotalEnergyConsumption()

    def __len__(self) -> int:
        """Returns the number of GPUs being tracked."""
        return len(self.gpus)


class NVIDIAGPUs(GPUs):
    """Represents a collection of NVIDIA GPUs for management and interaction, abstracting pyNVML calls and handling related exceptions.

    This class provides a high-level interface to interact with NVIDIA GPUs, taking into account the environment variable `CUDA_VISIBLE_DEVICES`
    for GPU visibility and allowing selective tracking of GPUs through `gpus_to_track`.

    `CUDA_VISIBLE_DEVICES` environment variable is respected if set. For example, if there are
    4 GPUs and `CUDA_VISIBLE_DEVICES=0,2`, only GPUs 0 and 2 are tracked. In this case, to access
    GPU of CUDA index 0, use the index 0, and for CUDA index 2, use the index 1.

    If gpus_to_track is specified, only the GPUs listed are set up. For continuation of the example above,
    if `gpus_to_track=[1]`, only GPU 2 of CUDA index 2 is tracked. In this case, to access GPU of CUDA index 2, use index 1.
    This is consistent with how `CUDA_VISIBLE_DEVICES` is conventionally handled.

    Parameters:
        gpu_indices (list[int], optional): A list of CUDA indices specifying which GPUs to track. Respects `CUDA_VISIBLE_DEVICES`.
        ensure_homogeneous (bool, optional): Ensures that all GPUs being tracked are homogeneous (i.e.,
            have the same output of `getName`).

    Attributes:
        visible_indices (list[int]): The list of GPU indices that are visible and considered for tracking,
            derived from `CUDA_VISIBLE_DEVICES` environment variable or system-wide visible devices if `CUDA_VISIBLE_DEVICES` is not set.
        gpus (dict): A dictionary mapping the tracked GPU indices to their respective `NVIDIAGPU` instances
            for easy access and management.

    Raises:
        ZeusBaseGPUError: An error specific to GPU operations, derived from the base exception class for GPU errors.
            This is raised when initialization fails due to an NVML error, with specifics provided by the NVML exception.
    """

    def _init_gpus(self, gpus_to_track: list[int] | None = None) -> None:
        # Must respect `CUDA_VISIBLE_DEVICES` if set
        if (visible_device := os.environ.get("CUDA_VISIBLE_DEVICES")) is not None:
            self.visible_indices = [int(idx) for idx in visible_device.split(",")]
        else:
            self.visible_indices = list(range(pynvml.nvmlDeviceGetCount()))

        # initialize all GPUs
        self.gpus = {}
        if gpus_to_track is not None:
            for gpu_num in gpus_to_track:
                self.gpus[gpu_num] = NVIDIAGPU(self.visible_indices[gpu_num])
        else:
            for index, gpu_num in enumerate(self.visible_indices):
                self.gpus[index] = NVIDIAGPU(gpu_num)

    def __init__(
        self, gpus_to_track: list[int] = None, ensure_homogeneous: bool = True
    ) -> None:
        """Initializes the NVIDIAGPUs instance, setting up tracking for specified NVIDIA GPUs.

        This involves initializing the NVML library to communicate with the NVIDIA driver and configuring which GPUs
        are visible based on the `CUDA_VISIBLE_DEVICES` environment variable and the optional `gpus_to_track`
        parameter. It ensures that only the specified GPUs (if any) are tracked and managed by this instance.

        Parameters:
            gpus_to_track (list[int], optional): Specifies the indices of the GPUs to be tracked. These indices
                should align with the visible GPU indices as determined by the system and the `CUDA_VISIBLE_DEVICES`
                environment variable. If None, all visible GPUs are tracked.
            ensure_homogeneous (bool, optional): If True, ensurea that all tracked GPUs have the same name (return value of pynvml.nvmlDeviceGetName).

        Raises:
            ZeusBaseGPUError: If an NVML error occurs, a specific GPU error is raised,
                encapsulating the original NVML error message for clearer debugging and error handling.
        """
        try:
            pynvml.nvmlInit()
            self._init_gpus(gpus_to_track)
            if ensure_homogeneous:
                self._ensure_homogeneous()
        except pynvml.NVMLError as e:
            exception_class = NVIDIAGPU._exception_map.get(e.value, ZeusBaseGPUError)
            raise exception_class(e.msg) from e

    def __del__(self) -> None:
        """Shuts down the NVIDIA GPU monitoring library to release resources and clean up."""
        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlShutdown()


class AMDGPUs(GPUs):
    """Represents a collection of AMD GPUs for management and interaction, abstracting amdsmi calls and handling related exceptions.

    This class provides a high-level interface to interact with AMD GPUs, taking into account the environment variable `ROCR_VISIBLE_DEVICES`
    for GPU visibility and allowing selective tracking of GPUs through `gpus_to_track`.

    `ROCR_VISIBLE_DEVICES` environment variable is respected if set. For example, if there are
    4 GPUs and `ROCR_VISIBLE_DEVICES=0,2`, only GPUs 0 and 2 are tracked. In this case, to access
    GPU of ROCR index 0, use the index 0, and for ROCR index 2, use the index 1.

    If gpus_to_track is specified, only the GPUs listed are set up. For continuation of the example above,
    if `gpus_to_track=[1]`, only GPU 2 of ROCR index 2 is tracked. In this case, to access GPU of ROCR index 2, use index 1.

    Parameters:
        gpus_to_track (list[int], optional): A list of integer indices specifying which GPUs to track.
            This list refers to the ROCR device indices as seen by the system. If not provided,
            all available GPUs will be tracked. The indices in this list are affected by the
            `ROCR_VISIBLE_DEVICES` environment variable, meaning that `gpus_to_track` should match
            against the visible devices, not necessarily against the system-wide device indices.
        ensure_homogeneous (bool, optional): Ensures that all GPUs being tracked are homogeneous (i.e.,
            of the same model and with the same specifications).

    Attributes:
        visible_indices (list[int]): The list of GPU indices that are visible and considered for tracking,
            derived from `ROCR_VISIBLE_DEVICES` environment variable or system-wide visible devices if `ROCR_VISIBLE_DEVICES` is not set.
        gpus (dict): A dictionary mapping the tracked GPU indices to their respective `AMDGPU` instances
            for easy access and management.

    Raises:
        ZeusBaseGPUError: An error specific to GPU operations, derived from the base exception class for GPU errors.
            This is raised when initialization fails due to an amdsmi error, with specifics provided by the amdsmi exception.
    """

    def _init_gpus(self, gpus_to_track: list[int] = None) -> None:
        # Must respect `ROCR_VISIBLE_DEVICES` if set
        if (visible_device := os.environ.get("ROCR_VISIBLE_DEVICES")) is not None:
            self.visible_indices = [int(idx) for idx in visible_device.split(",")]
        else:
            self.visible_indices = list(
                range(len(amdsmi.amdsmi_get_processor_handles()))
            )

        # initialize all GPUs
        self.gpus = {}
        if gpus_to_track is not None:
            for gpu_num in gpus_to_track:
                self.gpus[gpu_num] = AMDGPU(self.visible_indices[gpu_num])
        else:
            for index, gpu_num in enumerate(self.visible_indices):
                self.gpus[index] = AMDGPU(gpu_num)

    def __init__(
        self, gpus_to_track: list[int] = None, ensure_homogeneous: bool = True
    ) -> None:
        """Initializes the AMDGPUs instance, setting up tracking for specified AMD GPUs.

        This involves initializing the amdsmi library to communicate with the amdsmi driver and configuring which GPUs
        are visible based on the `ROCR_VISIBLE_DEVICES` environment variable and the optional `gpus_to_track`
        parameter. It ensures that only the specified GPUs (if any) are tracked and managed by this instance.

        Parameters:
            gpus_to_track (list[int], optional): Specifies the indices of the GPUs to be tracked. These indices
                should align with the visible GPU indices as determined by the system and the `ROCR_VISIBLE_DEVICES`
                environment variable. If None, all visible GPUs are tracked.
            ensure_homogeneous (bool, optional): If True, attempts to ensure that all tracked GPUs are of the same model
                and specifications.

        Raises:
            ZeusBaseGPUError: If an amdsmi error occurs, a specific GPU error is raised,
                encapsulating the original amdsmi error message for clearer debugging and error handling.
        """
        try:
            amdsmi.amdsmi_init()
            self._init_gpus(gpus_to_track)
            if ensure_homogeneous:
                self._ensure_homogeneous()
        except amdsmi.AmdSmiException as e:
            exception_class = AMDGPU._exception_map.get(e.value, ZeusBaseGPUError)
            raise exception_class(e.msg) from e

    def __del__(self) -> None:
        """Shuts down the AMD GPU monitoring library to release resources and clean up."""
        with contextlib.suppress(amdsmi.AmdSmiException):
            amdsmi.amdsmi_shut_down()  # Ignore error on shutdown. Neccessary for proper cleanup and test functionality


_gpus: GPUs | None = None


def get_gpus(gpus_to_track: list[int] = None, ensure_homogeneous: bool = True) -> GPUs:
    """Initialize (if called for the first time) and return a singleton GPU monitoring object for NVIDIA or AMD GPUs.

    This function attempts to initialize GPU monitoring using the pynvml library for NVIDIA GPUs
    first. If pynvml is not available or fails to initialize, it then tries to use the amdsmi
    library for AMD GPUs. If both attempts fail, it raises a ZeusErrorInit exception.

    For the GPU monitoring object, `CUDA_VISIBLE_DEVICES` environment variable is respected if set. For example, if there are
    4 GPUs and `CUDA_VISIBLE_DEVICES=0,2`, only GPUs 0 and 2 are tracked. In this case, to access
    GPU of CUDA index 0, use the index 0, and for CUDA index 2, use the index 1.

    If gpus_to_track is specified, only the GPUs listed are set up. For continuation of the example above,
    if `gpus_to_track=[1]`, only GPU 2 of CUDA index 2 is tracked. In this case, to access GPU of CUDA index 2, use index 1.

    Args:
        gpus_to_track (Optional[list[int]]): A list of integers representing the GPU indices
                                             to track. If None, all GPUs are tracked.
        ensure_homogeneous (bool, optional): If True, raises an error if the names of all GPUs are not homogeneous.

    Returns:
        GPUs: An instance of NVIDIAGPUs or AMDGPUs depending on the system's GPU.

    Raises:
        ZeusInitGPUError: If both NVIDIA and AMD GPU monitoring libraries fail to initialize.
    """
    global _gpus
    if _gpus is not None:
        return _gpus

    if pynvml_is_available():
        _gpus = NVIDIAGPUs(gpus_to_track, ensure_homogeneous)
        return _gpus
    elif amdsmi_is_available():
        _gpus = AMDGPUs(gpus_to_track, ensure_homogeneous)
        return _gpus
    else:
        raise ZeusInitGPUError(
            "Failed to initialize GPU monitoring for NVIDIA and AMD GPUs."
        )
    
    try:
        # Attempt to initialize NVIDIA GPUs
        # import pynvml <- this import fails when running tests
        pynvml.nvmlInit()
        _gpus = NVIDIAGPUs(gpus_to_track, ensure_homogeneous)
    except (ImportError, pynvml.NVMLError) as nvidia_error:
        try:
            # Attempt to initialize AMD GPUs
            import amdsmi

            amdsmi.amdsmi_init()
            _gpus = AMDGPUs(gpus_to_track, ensure_homogeneous)
        except (ImportError, amdsmi.AmdSmiLibraryException) as amd_error:
            # Both NVIDIA and AMD GPU monitoring libraries failed to initialize. Raise an exception.
            raise ZeusInitGPUError(
                f"Failed to initialize GPU monitoring for NVIDIA and AMD GPUs.\n"
                f"NVIDIA Error: {nvidia_error}\n"
                f"AMD Error: {amd_error}"
            ) from None
    return _gpus
