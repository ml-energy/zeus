# Abstracting away the GPU

"""
- Changing PL or freq requires SYS_ADMIN, and in production clusters, you can't give a ranndom application SYS_ADMIN
- Create a local server process w/ SYS_ADMIN, and have the application talk to the server. Only SYS_ADMIN-required methods

First step: Abstracting away the GPU
- Current state: Call NVML directly (won't work in production)

Usages of pynvml:
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
minmax = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
pynvml.nvmlShutdown()
pynvml.nvmlDeviceGetCount()
pynvml.nvmlDeviceSetPowerManagementLimit(
pynvml.nvmlDeviceSetPersistenceMode(handle, pynvml.NVML_FEATURE_ENABLED)
pynvml.nvmlDeviceGetHandleByIndex(index)
pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, max_mem_freq)
pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)

with contextlib.suppress(pynvml.NVMLError_NotSupported):  # type: ignore
  87:             pynvml.nvmlDeviceResetMemoryLockedClocks(handle)
  88:         pynvml.nvmlDeviceResetGpuLockedClocks(handle)

"""

from __future__ import annotations

import abc
import pynvml
import os

from exception import ZeusBaseGPUError


import amdsmi

class GPU(abc.ABC):
    def __init__(self, gpu_index: int) -> None:
        self.gpu_index = gpu_index

    @abc.abstractmethod
    def get_total_energy_consumption(self) -> float:
        pass

    @abc.abstractmethod
    def set_power_limit(self, value: int) -> None:
        pass


""" NVIDIA GPUs """

class NativeNVIDIAGPU(GPU):
    """Alternatve names: DirectNVIDIAGPU, LocalNVIDIAGPU, NativeNVIDIAGPU, IntegratedNVIDIAGPU"""
    def __init__(self, gpu_index: int) -> None:
        super().__init__(gpu_index)
    
    def GetPowerManagementLimitConstraints(self) -> tuple[int, int]:
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        return pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)

    def SetPersistenceMode(self) -> None:
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        pynvml.nvmlDeviceSetPersistenceMode(handle, pynvml.NVML_FEATURE_ENABLED)

    def SetPowerManagementLimit(self, value: int) -> None:
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        pynvml.nvmlDeviceSetPowerManagementLimit(handle, value)
    
    def GetSupportedGraphicsClocks(self) -> list[int]:
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        max_mem_freq = max(pynvml.nvmlDeviceGetSupportedMemoryClocks(handle))
        return pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, max_mem_freq)



class RemoteNVIDIAGPU(NativeNVIDIAGPU):
    def __init__(self, gpu_index: int, server_address: str) -> None:
        super().__init__(gpu_index)
        self.server_address = server_address

    def set_power_limit(self, value: int) -> None:
        # Call server since SYS_ADMIN is required
        pass
    def set_freq_mem(self, value: int) -> None:
        # Call server since SYS_ADMIN is required
        pass
    def set_freq_core(self, value: int) -> None:
        # Call server since SYS_ADMIN is required
        pass


""" AMD GPUs """

class NativeAMDGPU(GPU):
    def __init__(self, gpu_index: int) -> None:
        super().__init__(gpu_index)
    
    def GetPowerManagementLimitConstraints(self) -> tuple[int, int]:
        handle = amdsmi.amdsmi_get_processor_handles()[self.gpu_index]
        info = amdsmi.amdsmi_get_power_cap_info(handle)
        return (info.min_power_cap, info.max_power_cap) 

    def SetPersistenceMode(self) -> None:
        handle = amdsmi.amdsmi_get_processor_handles()[self.gpu_index]
        profile = ... # TODO: find out correct profile, need deep dive on docs
        amdsmi.amdsmi_set_gpu_power_profile(handle, 0, profile)
    
    def SetPowerManagementLimit(self, value: int) -> None:
        handle = amdsmi.amdsmi_get_processor_handles()[self.gpu_index]
        """
        Input parameters:
        processor_handle handle for the given device
        sensor_ind a 0-based sensor index. Normally, this will be 0. If a device has more than one sensor, it could be greater than 0
        cap int that indicates the desired power cap, in microwatts
        """
        amdsmi.amdsmi_set_power_cap(handle, sensor_id=0, cap=value)
    
    def GetSupportedGraphicsClocks(self) -> list[int]:
        handle = amdsmi.amdsmi_get_processor_handles()[self.gpu_index]
        num_supported, current, frequency = amdsmi.amdsmi_get_clk_freq(handle, clk_type=amdsmi.AmdSmiClkType.SYS) #TODO: Figure out correct clk_type
        """ frequency; List of frequencies, only the first num_supported frequencies are valid"""
        return frequency[:num_supported]

class RemoteAMDGPU(NativeAMDGPU):
    pass



# Managing all GPUs - Base Abstract Class
class GPUs(abc.ABC):
    # Abstract class for managing all GPUs
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def Init(self) -> None:
        pass

    @abc.abstractmethod
    def Shutdown(self) -> None:
        pass


"""

should NVIDIAGPUs track all GPUs or only the ones visible to CUDA? I don't see the need
to track all GPUs, but maybe there are some edge cases where it's useful.

should AMD GPUs look at CUDA_VISIBLE_DEVICES? no just ROCM_VISIBLE_DEVICES right?

in the code there is a lot of self.gpu_handles. Should be removed and abstracted away?

why is nvmlDeviceSetPersistenceMode necessary?

no throw destructor?

different subclasses for different types of GPUs? ex. NVIDIA volta vs ampere?
"""

# NVIDIA GPUs
class NVIDIAGPUs(GPUs):
    def __init__(self, ensure_homogeneuous: bool = True) -> None:
        
        pynvml.nvmlInit()
        

        # Must respect `CUDA_VISIBLE_DEVICES` if set
        if (visible_device := os.environ.get("CUDA_VISIBLE_DEVICES")) is not None:
            self.visible_indices = [int(idx) for idx in visible_device.split(",")]
        else:
            self.visible_indices = list(range(pynvml.nvmlDeviceGetCount()))
            
        # initialize all GPUs
        self.gpus = [NativeNVIDIAGPU(index) for index in self.visible_indices]

    def GetPowerManagementLimitConstraints(self, index: int) -> tuple[int, int]:
        """ 
        zeus/run/master.pu line 106
        
        Analogous to handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        minmax = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)  # unit: mW"""
        return self.gpus[index].GetPowerManagementLimitConstraints()

    def SetPersistenceMode(self, index: int) -> None:

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
        self.gpus[index].SetPersistenceMode()
    
    def SetPowerManagementLimit(self, index: int, value: int) -> None:
        self.gpus[index].SetPowerManagementLimit(value)

    def GetSupportedGraphicsClocks(self, index: int) -> list[int]:
        self.gpus[index].GetSupportedGraphicsClocks()

    def __del__(self) -> None:
        pynvml.nvmlShutdown()


#AMD GPUs

class AMDGPUs(GPUs):
    def __init__(self, ensure_homogeneuous: bool = True) -> None:
        amdsmi.amdsmi_init()

        # Must respect `ROCR_VISIBLE_DEVICES` if set
        if (visible_device := os.environ.get("ROCR_VISIBLE_DEVICES")) is not None:
            self.visible_indices = [int(idx) for idx in visible_device.split(",")]
        else:
            self.visible_indices = list(range(pynvml.nvmlDeviceGetCount()))
            
        # initialize all GPUs
        self.gpus = [NativeAMDGPU(index) for index in self.visible_indices]

    def GetPowerManagementLimitConstraints(self, index: int) -> tuple[int, int]:
        """ 
        zeus/run/master.pu line 106
        
        Analogous to handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        minmax = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)  # unit: mW"""
        return self.gpus[index].GetPowerManagementLimitConstraints()

    def SetPowerManagementLimit(self, index: int, value: int) -> None:
        self.gpus[index].SetPowerManagementLimit(value)
    
    def GetSupportedGraphicsClocks(self, index: int) -> list[int]:
        self.gpus[index].GetSupportedGraphicsClocks()

    def __del__(self) -> None:
        amdsmi.amdsmi_shut_down()



# Ensure only one instance of GPUs is created
def get_gpus() -> GPUs:
    return NVIDIAGPUs() if pynvml.is_initialized() else AMDGPUs()

GPUManager = get_gpus()



""" EXCEPTION WRAPPERS """

class ZeusGPUErrorUninit(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for NVML_ERROR_UNINITIALIZED."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)

class ZeusGPUErrorInvalidArg(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for NVML_ERROR_UNINITIALIZED."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)