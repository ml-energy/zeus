"""GPU management module for Zeus."""

from __future__ import annotations
from typing import TYPE_CHECKING

import abc
import os

# from exception import ZeusBaseGPUError
if TYPE_CHECKING:
    import pynvml
    import amdsmi

import sys

UNIT_TESTING = 'pytest' in sys.modules
if UNIT_TESTING:
    import pynvml

# import pynvml

from zeus.util import cuda_sync

class GPU(abc.ABC):
    def __init__(self, gpu_index: int) -> None:
        self.gpu_index = gpu_index


""" NVIDIA GPUs """

class NVIDIAGPU(GPU):
    """Alternatve names: DirectNVIDIAGPU, LocalNVIDIAGPU, NativeNVIDIAGPU, IntegratedNVIDIAGPU"""
    def __init__(self, gpu_index: int) -> None:
        super().__init__(gpu_index)
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
    
    def getPowerManagementLimitConstraints(self) -> tuple[int, int]:
        return pynvml.nvmlDeviceGetPowerManagementLimitConstraints(self.handle)

    def setPersistenceMode(self) -> None:
        # TODO(JW): Check SYS_ADMIN permissions and error with an explanation.
        pynvml.nvmlDeviceSetPersistenceMode(self.handle, pynvml.NVML_FEATURE_ENABLED)

    def setPowerManagementLimit(self, value: int = None) -> None:
        """if defualt is True, set to default value, else set to value"""
        if value is None:
            pynvml.nvmlDeviceSetPowerManagementLimit(self.handle, pynvml.nvmlDeviceGetPowerManagementDefaultLimit(self.handle))
        else:
            pynvml.nvmlDeviceSetPowerManagementLimit(self.handle, value)
    
    def setMemoryLockedClocks(self, minMemClockMHz: int, maxMemClockMHz: int) -> None:
        pynvml.nvmlDeviceSetMemoryLockedClocks(self.handle, minMemClockMHz, maxMemClockMHz)
    
    def getName(self) -> str:
        return pynvml.nvmlDeviceGetName(self.handle)
    
    def getSupportedMemoryClocks(self) -> list[int]:
        return pynvml.nvmlDeviceGetSupportedMemoryClocks(self.handle)
    
    def getSupportedGraphicsClocks(self, freq: int) -> list[int]:
        return pynvml.nvmlDeviceGetSupportedGraphicsClocks(self.handle, freq)
    
    def setGpuLockedClocks(self, index: int, minMemClockMHz: int, maxMemClockMHz: int) -> None:
        pynvml.nvmlDeviceSetGpuLockedClocks(self.handle, minMemClockMHz, maxMemClockMHz)
    
    def resetMemoryLockedClocks(self) -> None:
        pynvml.nvmlDeviceResetMemoryLockedClocks(self.handle)
    
    def resetGpuLockedClocks(self) -> None:
        pynvml.nvmlDeviceResetGpuLockedClocks(self.handle)
    
    def getPowerUsage(self) -> int:
        return pynvml.nvmlDeviceGetPowerUsage(self.handle)
    
    def supportsGetTotalEnergyConsumption(self) -> bool:
        # NVIDIA GPUs Volta or newer support this method
        return (
            pynvml.nvmlDeviceGetArchitecture(self.handle)
            >= pynvml.NVML_DEVICE_ARCH_VOLTA
        )
    
    def getTotalEnergyConsumption(self) -> int:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)


class UnprivilegedNVIDIAGPU(NVIDIAGPU):
    def __init__(self, gpu_index: int) -> None:
        super().__init__(gpu_index)


""" AMD GPUs """

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



# Managing all GPUs - Base Abstract Class
class GPUs(abc.ABC):
    # Abstract class for managing all GPUs
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __del__(self) -> None:
        pass

    def getPowerManagementLimitConstraints(self, index: int) -> tuple[int, int]:
        """ 
        zeus/run/master.pu line 106
        
        Analogous to handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        minmax = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)  # unit: mW"""
        return self.gpus[index].getPowerManagementLimitConstraints()

    def setPersistenceMode(self, index: int) -> None:

        """
        Can be changed to take more than one index

        dataloader.py ln 423
        for index in range(self.world_size):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            # Set persistent mode.
            # TODO(JW): Check SYS_ADMIN permissions and error with an explanation.
            pynvml.nvmlDeviceSetPersistenceMode(handle, pynvml.NVML_FEATURE_ENABLED)
            self.gpu_handles.append(handle)
        
        """
        self.gpus[index].setPersistenceMode()
    
    def setPowerManagementLimit(self, index: int, value: int = None) -> None:
        """Sets the power management limit to a specified value or to the default if no value is provided."""
        self.gpus[index].setPowerManagementLimit(value)
    
    def setMemoryLockedClocks(self, index: int, minMemClockMHz: int, maxMemClockMHz: int) -> None:
        self.gpus[index].setMemoryLockedClocks(minMemClockMHz, maxMemClockMHz)

    def getSupportedMemoryClocks(self, index: int) -> list[int]:
        self.gpus[index].getSupportedMemoryClocks()

    def getSupportedGraphicsClocks(self, index: int, freq: int) -> list[int]:
        self.gpus[index].getSupportedGraphicsClocks(freq)
    
    def getName(self, index: int) -> str:
        return self.gpus[index].getName()
    
    def __len__(self) -> int:
        return len(self.gpus)
    
    def setGpuLockedClocks(self, index: int, minMemClockMHz: int, maxMemClockMHz: int) -> None:
        self.gpus[index].setGpuLockedClocks(minMemClockMHz, maxMemClockMHz)
    
    def resetMemoryLockedClocks(self, index: int) -> None:
        self.gpus[index].resetMemoryLockedClocks()

    def resetGpuLockedClocks(self, index: int) -> None:
        self.gpus[index].resetGpuLockedClocks()
    
    def getPowerUsage(self, index: int) -> int:
        return self.gpus[index].getPowerUsage()
    
    def supportsGetTotalEnergyConsumption(self, index: int) -> bool:
        return self.gpus[index].supportsGetTotalEnergyConsumption()
    
    def getTotalEnergyConsumption(self, index: int) -> int:
        return self.gpus[index].getTotalEnergyConsumption()
    
    def sync(self, index: int) -> None:
        cuda_sync(index) # cuda_sync takes in re-indexed cuda index, not nvml index

# NVIDIA GPUs
class NVIDIAGPUs(GPUs):
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
        pynvml.nvmlInit()
        self.init_GPUs(gpus_to_track)

    def __del__(self) -> None:
        try:
            pynvml.nvmlShutdown()
        except:
            pass


#AMD GPUs

class AMDGPUs(GPUs):
    def init_GPUs(self) -> None:
        # Must respect `ROCR_VISIBLE_DEVICES` if set
        if (visible_device := os.environ.get("ROCR_VISIBLE_DEVICES")) is not None:
            self.visible_indices = [int(idx) for idx in visible_device.split(",")]
        else:
            self.visible_indices = list(range(pynvml.nvmlDeviceGetCount()))
            
        # initialize all GPUs
        self.gpus = [AMDGPU(index) for index in self.visible_indices]

    def __init__(self, ensure_homogeneuous: bool = True) -> None:
        amdsmi.amdsmi_init()

    def __del__(self) -> None:
        amdsmi.amdsmi_shut_down()


def get_gpus(gpus_to_track: list[int] = None) -> GPUs:

    """
    Initialize and return a singleton GPU monitoring object for NVIDIA or AMD GPUs.

    This function attempts to initialize GPU monitoring using the pynvml library for NVIDIA GPUs
    first. If pynvml is not available or fails to initialize, it then tries to use the amdsmi 
    library for AMD GPUs. If both attempts fail, it raises a ZeusErrorInit exception.
    
    This function supports tracking specific GPUs by passing their indices in `gpus_to_track`. 

    Args:
        gpus_to_track (Optional[list[int]]): A list of integers representing the GPU indices
                                             to track. If None, all GPUs are tracked.

    Returns:
        GPUs: An instance of NVIDIAGPUs or AMDGPUs depending on the system's GPU.
    """
    global _gpus
    if _gpus is not None:
        return _gpus

    try:
        import pynvml
        pynvml.nvmlInit()
        _gpus = NVIDIAGPUs(gpus_to_track)
    except ImportError:
        try:
            import amdsmi
            amdsmi.amdsmi_init()
            _gpus = AMDGPUs()
        except ImportError:
            return None
    return _gpus

""" EXCEPTION WRAPPERS """

# class ZeusGPUErrorUninit(ZeusBaseGPUError):
#     """Zeus GPU exception class Wrapper for NVML_ERROR_UNINITIALIZED."""

#     def __init__(self, message: str) -> None:
#         """Initialize Zeus Exception."""
#         super().__init__(message)

# class ZeusGPUErrorInvalidArg(ZeusBaseGPUError):
#     """Zeus GPU exception class Wrapper for NVML_ERROR_UNINITIALIZED."""

#     def __init__(self, message: str) -> None:
#         """Initialize Zeus Exception."""
#         super().__init__(message)