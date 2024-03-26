"""GPU management module for Zeus."""

from __future__ import annotations
from typing import TYPE_CHECKING
import functools

import abc
import os

from zeus.device.exception import ZeusBaseGPUError
if TYPE_CHECKING:
    import pynvml
    import amdsmi

import sys

import pynvml

from zeus.util import cuda_sync

""" EXCEPTION WRAPPERS """

class ZeusInitGPUError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for ImportError or Failed to Initialize GPU libraries."""

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


class GPU(abc.ABC):
    def __init__(self, gpu_index: int) -> None:
        self.gpu_index = gpu_index


""" NVIDIA GPUs """

class NVIDIAGPU(GPU):
    """"""
    _exception_map = {
            pynvml.NVML_ERROR_UNINITIALIZED : ZeusInitGPUError,
            pynvml.NVML_ERROR_INVALID_ARGUMENT : ZeusInvalidArgGPUError,
            pynvml.NVML_ERROR_NOT_SUPPORTED : ZeusNotSupportedGPUError,
            pynvml.NVML_ERROR_NO_PERMISSION : ZeusNoPermissionGPUError,
            pynvml.NVML_ERROR_ALREADY_INITIALIZED : ZeusAlreadyInitializedGPUError,
            pynvml.NVML_ERROR_NOT_FOUND : ZeusNotFoundGPUError,
            pynvml.NVML_ERROR_INSUFFICIENT_SIZE : ZeusInsufficientSizeGPUError,
            pynvml.NVML_ERROR_INSUFFICIENT_POWER : ZeusInsufficientPowerGPUError,
            pynvml.NVML_ERROR_DRIVER_NOT_LOADED : ZeusDriverErrorGPUError,
            pynvml.NVML_ERROR_TIMEOUT : ZeusTimeoutGPUError,
            pynvml.NVML_ERROR_IRQ_ISSUE : ZeusIRQErrorGPUError,
            pynvml.NVML_ERROR_LIBRARY_NOT_FOUND : ZeusLibraryNotFoundGPUError,
            pynvml.NVML_ERROR_FUNCTION_NOT_FOUND : ZeusFunctionNotFoundGPUError,
            pynvml.NVML_ERROR_CORRUPTED_INFOROM : ZeusCorruptedInfoROMGPUError,
            pynvml.NVML_ERROR_GPU_IS_LOST : ZeusLostGPUGPUError,
            pynvml.NVML_ERROR_RESET_REQUIRED : ZeusResetRequiredGPUError,
            pynvml.NVML_ERROR_OPERATING_SYSTEM : ZeusOperatingSystemGPUError,
            pynvml.NVML_ERROR_LIB_RM_VERSION_MISMATCH : ZeusLibRMVersionMismatchGPUError,
            pynvml.NVML_ERROR_MEMORY : ZeusMemoryGPUError,
            pynvml.NVML_ERROR_UNKNOWN : ZeusUnknownGPUError
    }

    def handle_nvml_errors(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except pynvml.NVMLError as e:
                exception_class = NVIDIAGPU._exception_map.get(e.value, ZeusBaseGPUError)
                raise exception_class(e.msg) from e
        return wrapper

    def __init__(self, gpu_index: int) -> None:
        super().__init__(gpu_index)
        self._get_handle()

    @handle_nvml_errors
    def _get_handle(self):
        try:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except pynvml.NVMLError as e:
                exception_class = NVIDIAGPU._exception_map.get(e.value, ZeusBaseGPUError)
                raise exception_class(e.msg) from e
    
    @handle_nvml_errors
    def getPowerManagementLimitConstraints(self) -> tuple[int, int]:
        return pynvml.nvmlDeviceGetPowerManagementLimitConstraints(self.handle)

    @handle_nvml_errors
    def setPersistenceMode(self) -> None:
        # TODO(JW): Check SYS_ADMIN permissions and error with an explanation.
        pynvml.nvmlDeviceSetPersistenceMode(self.handle, pynvml.NVML_FEATURE_ENABLED)

    @handle_nvml_errors
    def setPowerManagementLimit(self, value: int = None) -> None:
        """if defualt is True, set to default value, else set to value"""
        if value is None:
            pynvml.nvmlDeviceSetPowerManagementLimit(self.handle, pynvml.nvmlDeviceGetPowerManagementDefaultLimit(self.handle))
        else:
            pynvml.nvmlDeviceSetPowerManagementLimit(self.handle, value)
    
    @handle_nvml_errors
    def setMemoryLockedClocks(self, minMemClockMHz: int, maxMemClockMHz: int) -> None:
        pynvml.nvmlDeviceSetMemoryLockedClocks(self.handle, minMemClockMHz, maxMemClockMHz)
    
    @handle_nvml_errors
    def getName(self) -> str:
        return pynvml.nvmlDeviceGetName(self.handle)
    
    @handle_nvml_errors
    def getSupportedMemoryClocks(self) -> list[int]:
        return pynvml.nvmlDeviceGetSupportedMemoryClocks(self.handle)
    
    @handle_nvml_errors
    def getSupportedGraphicsClocks(self, freq: int) -> list[int]:
        return pynvml.nvmlDeviceGetSupportedGraphicsClocks(self.handle, freq)
    
    @handle_nvml_errors
    def setGpuLockedClocks(self, index: int, minMemClockMHz: int, maxMemClockMHz: int) -> None:
        pynvml.nvmlDeviceSetGpuLockedClocks(self.handle, minMemClockMHz, maxMemClockMHz)
    
    @handle_nvml_errors
    def resetMemoryLockedClocks(self) -> None:
        pynvml.nvmlDeviceResetMemoryLockedClocks(self.handle)
    
    @handle_nvml_errors
    def resetGpuLockedClocks(self) -> None:
        pynvml.nvmlDeviceResetGpuLockedClocks(self.handle)
    
    @handle_nvml_errors
    def getPowerUsage(self) -> int:
        return pynvml.nvmlDeviceGetPowerUsage(self.handle)
    
    @handle_nvml_errors
    def supportsGetTotalEnergyConsumption(self) -> bool:
        # NVIDIA GPUs Volta or newer support this method
        return (
            pynvml.nvmlDeviceGetArchitecture(self.handle)
            >= pynvml.NVML_DEVICE_ARCH_VOLTA
        )
    
    @handle_nvml_errors
    def getTotalEnergyConsumption(self) -> int:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)


class UnprivilegedNVIDIAGPU(NVIDIAGPU):
    def __init__(self, gpu_index: int) -> None:
        super().__init__(gpu_index)


class AMDGPU(GPU):
    def __init__(self, gpu_index: int) -> None:
        super().__init__(gpu_index)
        self.handle = amdsmi.amdsmi_get_processor_handles()[self.gpu_index]
    
    def getPowerManagementLimitConstraints(self) -> tuple[int, int]:
        info = amdsmi.amdsmi_get_power_cap_info(self.handle)
        return (info.min_power_cap, info.max_power_cap)  

    def setPersistenceMode(self) -> None:
        profile = ... # TODO: find out correct profile, need deep dive on docs
        amdsmi.amdsmi_set_gpu_power_profile(self.handle, 0, profile)
    
    def setPowerManagementLimit(self, value: int = None) -> None:
        """
        Input parameters:
        processor_handle handle for the given device
        sensor_ind a 0-based sensor index. Normally, this will be 0. If a device has more than one sensor, it could be greater than 0
        cap int that indicates the desired power cap, in microwatts
        """
        if value is None:
            info = amdsmi.amdsmi_get_power_cap_info(self.handle)
            amdsmi.amdsmi_set_power_cap(self.handle, sensor_id=0, cap=info.default_power_cap)
        else:
            amdsmi.amdsmi_set_power_cap(self.handle, sensor_id=0, cap=value)
    
    def setMemoryLockedClocks(self, minMemClockMHz: int, maxMemClockMHz: int) -> None:
        amdsmi.amdsmi_set_gpu_clk_range(self.handle, minMemClockMHz, maxMemClockMHz, clk_type= amdsmi.AMDSMI_CLK_TYPE_SYS)

    def getName(self) -> str:
        market_name, vendor_id, device_id, rev_id, asic_serial = amdsmi.amdsmi_get_gpu_asic_info(self.handle) #TODO: Does this return correct string
        return market_name
    
    def getSupportedMemoryClocks(self) -> list[int]:
        num_supported, current, frequency = amdsmi.amdsmi_get_clk_freq(self.handle, clk_type=amdsmi.AMDSMI_CLK_TYPE_SYS) #TODO: Figure out correct clk_type
        """ frequency; List of frequencies, only the first num_supported frequencies are valid"""
        return frequency[:num_supported]

    def getSupportedGraphicsClocks(self, freq: int) -> list[int]:
        # TODO: what does 137-140 in optimizer.py do?
        pass

    def setGpuLockedClocks(self, index: int, minMemClockMHz: int, maxMemClockMHz: int) -> None:
        amdsmi.amdsmi_set_gpu_clk_range(self.handle, minMemClockMHz, maxMemClockMHz, clk_type=amdsmi.AMDSMI_CLK_TYPE_GFX)

    def resetMemoryLockedClocks(self) -> None:
        amdsmi.amdsmi_reset_gpu_clk(self.handle, clk_type=amdsmi.AMDSMI_CLK_TYPE_SYS) # TODO: check docs

    def resetGpuLockedClocks(self) -> None:
        amdsmi.amdsmi_reset_gpu_clk(self.handle, clk_type=amdsmi.AMDSMI_CLK_TYPE_GFX) # TODO: check docs
    
    def getPowerUsage(self) -> int:
        return None # TODO: figure out how to get power usage

class UnprivilegedAMDGPU(AMDGPU):
    pass


class GPUs(abc.ABC):
    """
    An abstract base class for managing and interacting with GPUs. This class defines the
    essential interface and common functionality for GPU management, including power management,
    clock settings, and information retrieval. Implementations of this class should provide
    specific methods to interact with different types of GPUs or GPU libraries (e.g., NVML for NVIDIA GPUs).

    Methods defined in this class are abstract and must be implemented by subclasses. These
    methods provide a framework for initializing and cleaning up GPU resources, along with
    a set of operations commonly used for GPU management."""

    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __del__(self) -> None:
        pass

    def getPowerManagementLimitConstraints(self, index: int) -> tuple[int, int]:
        """Returns the minimum and maximum power management limits for the specified GPU."""
        return self.gpus[index].getPowerManagementLimitConstraints()

    def setPersistenceMode(self, index: int) -> None:
        """Enables persistence mode for the specified GPU."""
        self.gpus[index].setPersistenceMode()
    
    def setPowerManagementLimit(self, index: int, value: int = None) -> None:
        """Sets the power management limit for the specified GPU to the given value. If no value is provided, the default limit is set."""
        self.gpus[index].setPowerManagementLimit(value)
    
    def setMemoryLockedClocks(self, index: int, minMemClockMHz: int, maxMemClockMHz: int) -> None:
        """Locks the memory clock of the specified GPU to a range defined by the minimum and
            maximum memory clock frequencies."""
        self.gpus[index].setMemoryLockedClocks(minMemClockMHz, maxMemClockMHz)

    def getSupportedMemoryClocks(self, index: int) -> list[int]:
        """Returns a list of supported memory clock frequencies for the specified GPU."""
        self.gpus[index].getSupportedMemoryClocks()

    def getSupportedGraphicsClocks(self, index: int, freq: int) -> list[int]:
        """Returns a list of supported graphics clock frequencies for the specified GPU at
            a given frequency."""
        self.gpus[index].getSupportedGraphicsClocks(freq)
    
    def getName(self, index: int) -> str:
        """Returns the name of the specified GPU."""
        return self.gpus[index].getName()
    
    def __len__(self) -> int:
        """Returns the number of GPUs being tracked."""
        return len(self.gpus)
    
    def setGpuLockedClocks(self, index: int, minMemClockMHz: int, maxMemClockMHz: int) -> None:
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
    
    def sync(self, index: int) -> None:
        """Synchronizes the specified GPU, ensuring all previous commands have been completed."""
        cuda_sync(index) # cuda_sync takes in re-indexed cuda index, not nvml index

class NVIDIAGPUs(GPUs):
    """
    Represents a collection of NVIDIA GPUs for management and interaction, abstracting 
    pyNVML calls and handling related exceptions. This class provides a high-level interface 
    to interact with NVIDIA GPUs, taking into account the environment variable `CUDA_VISIBLE_DEVICES` 
    for GPU visibility and allowing selective tracking of GPUs through `gpus_to_track`.

    `CUDA_VISIBLE_DEVICES` environment variable is respected if set. For example, if there are
    4 GPUs and `CUDA_VISIBLE_DEVICES=0,2`, only GPUs 0 and 2 are tracked. In this case, to access
    GPU of CUDA index 0, use the index 0, and for CUDA index 2, use the index 1.

    If gpus_to_track is specified, only the GPUs listed are set up. For continuation of the example above,
    if `gpus_to_track=[1]`, only GPU 2 of CUDA index 2 is tracked. In this case, to access GPU of CUDA index 2, use index 1.

    Parameters:
        gpus_to_track (list[int], optional): A list of integer indices specifying which GPUs to track.
            This list refers to the CUDA device indices as seen by the system. If not provided,
            all available GPUs will be tracked. The indices in this list are affected by the 
            `CUDA_VISIBLE_DEVICES` environment variable, meaning that `gpus_to_track` should match
            against the visible devices, not necessarily against the system-wide device indices.
        ensure_homogeneous (bool, optional): Ensures that all GPUs being tracked are homogeneous (i.e., 
            of the same model and with the same specifications).

    Attributes:
        visible_indices (list[int]): The list of GPU indices that are visible and considered for tracking, 
            derived from `CUDA_VISIBLE_DEVICES` environment variable or system-wide visible devices if `CUDA_VISIBLE_DEVICES` is not set.
        gpus (dict): A dictionary mapping the tracked GPU indices to their respective `NVIDIAGPU` instances 
            for easy access and management.

    Raises:
        ZeusBaseGPUError: An error specific to GPU operations, derived from the base exception class for GPU errors.
            This is raised when initialization fails due to an NVML error, with specifics provided by the NVML exception.
    """
    def init_GPUs(self, gpus_to_track: list[int] = None) -> None:
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
        

    def __init__(self, gpus_to_track: list[int] = None, ensure_homogeneuous: bool = True) -> None:
        """
        Initializes the NVIDIAGPUs instance, setting up tracking for specified NVIDIA GPUs. This involves 
        initializing the NVML library to communicate with the NVIDIA driver and configuring which GPUs 
        are visible based on the `CUDA_VISIBLE_DEVICES` environment variable and the optional `gpus_to_track` 
        parameter. It ensures that only the specified GPUs (if any) are tracked and managed by this instance.

        Parameters:
            gpus_to_track (list[int], optional): Specifies the indices of the GPUs to be tracked. These indices 
                should align with the visible GPU indices as determined by the system and the `CUDA_VISIBLE_DEVICES` 
                environment variable. If None, all visible GPUs are tracked.
            ensure_homogeneous (bool, optional): If True, attempts to ensure that all tracked GPUs are of the same model 
                and specifications.

        Raises:
            ZeusBaseGPUError: If an NVML error occurs, a specific GPU error is raised, 
                encapsulating the original NVML error message for clearer debugging and error handling.
        """

        try:
            pynvml.nvmlInit()
            self.init_GPUs(gpus_to_track)
        except pynvml.NVMLError as e:
            exception_class = NVIDIAGPU._exception_map.get(e.value, ZeusBaseGPUError)
            raise exception_class(e.msg) from e

    def __del__(self) -> None:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as e:
            pass # Ignore error on shutdown. Neccessary for proper cleanup and test functionality.

class AMDGPUs(GPUs):
    """
    Represents a collection of AMD GPUs for management and interaction, abstracting 
    amdsmi calls and handling related exceptions. This class provides a high-level interface 
    to interact with AMD GPUs, taking into account the environment variable `ROCR_VISIBLE_DEVICES` 
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
    def init_GPUs(self, gpus_to_track: list[int] = None) -> None:
        # Must respect `ROCR_VISIBLE_DEVICES` if set
        if (visible_device := os.environ.get("ROCR_VISIBLE_DEVICES")) is not None:
            self.visible_indices = [int(idx) for idx in visible_device.split(",")]
        else:
            self.visible_indices = list(range(pynvml.nvmlDeviceGetCount()))

        # initialize all GPUs
        self.gpus = {}
        if gpus_to_track is not None:
            for gpu_num in gpus_to_track:
                self.gpus[gpu_num] = AMDGPU(self.visible_indices[gpu_num])
        else:
            for index, gpu_num in enumerate(self.visible_indices):
                self.gpus[index] = AMDGPU(gpu_num)

    def __init__(self, gpus_to_track: list[int] = None, ensure_homogeneuous: bool = True) -> None:
        """
        Initializes the AMDGPUs instance, setting up tracking for specified AMD GPUs. This involves 
        initializing the amdsmi library to communicate with the amdsmi driver and configuring which GPUs 
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
        amdsmi.amdsmi_init()
        self.init_GPUs()

    def __del__(self) -> None:
        try:
            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiException as e:
            pass # Ignore error on shutdown. Neccessary for proper cleanup and test functionality.

_gpus = None
def get_gpus(gpus_to_track: list[int] = None, ensure_homogeneuous: bool = True) -> GPUs:
    """
    Initialize and return a singleton GPU monitoring object for NVIDIA or AMD GPUs.

    This function attempts to initialize GPU monitoring using the pynvml library for NVIDIA GPUs
    first. If pynvml is not available or fails to initialize, it then tries to use the amdsmi 
    library for AMD GPUs. If both attempts fail, it raises a ZeusErrorInit exception.

    `CUDA_VISIBLE_DEVICES` environment variable is respected if set. For example, if there are
    4 GPUs and `CUDA_VISIBLE_DEVICES=0,2`, only GPUs 0 and 2 are tracked. In this case, to access
    GPU of CUDA index 0, use the index 0, and for CUDA index 2, use the index 1.

    If gpus_to_track is specified, only the GPUs listed are set up. For continuation of the example above,
    if `gpus_to_track=[1]`, only GPU 2 of CUDA index 2 is tracked. In this case, to access GPU of CUDA index 2, use index 1.

    Args:
        gpus_to_track (Optional[list[int]]): A list of integers representing the GPU indices
                                             to track. If None, all GPUs are tracked.
        ensure_homogeneous (bool, optional): If True, attempts to ensure that all tracked GPUs are of the same model 
                and specifications.

    Returns:
        GPUs: An instance of NVIDIAGPUs or AMDGPUs depending on the system's GPU.

    Raises:
        ZeusInitGPUError: If both NVIDIA and AMD GPU monitoring libraries fail to initialize.
    """
    global _gpus
    if _gpus is not None:
        return _gpus
    try:
        # Attempt to initialize NVIDIA GPUs
        # import pynvml <- this import fails when running tests
        pynvml.nvmlInit()
        _gpus = NVIDIAGPUs(gpus_to_track, ensure_homogeneuous)
    except (ImportError, pynvml.NVMLError) as nvidia_error:
        try:
            # Attempt to initialize AMD GPUs
            import amdsmi
            amdsmi.amdsmi_init()
            _gpus = AMDGPUs(gpus_to_track, ensure_homogeneuous)
        except (ImportError, amdsmi.AmdSmiLibraryException) as amd_error:
            # Both NVIDIA and AMD GPU monitoring libraries failed to initialize. Raise an exception.
            raise ZeusInitGPUError(
                f"Failed to initialize GPU monitoring for NVIDIA and AMD GPUs.\n"
                f"NVIDIA Error: {nvidia_error}\n"
                f"AMD Error: {amd_error}"
            ) from None
    return _gpus

