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


# import amdsmi

class GPU(abc.ABC):
    def __init__(self, gpu_index: int) -> None:
        self.gpu_index = gpu_index


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

    def SetPowerManagementLimit(self, value: int, default: bool = False) -> None:
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        if default:
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle))
        else:
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, value)
    
    def SetMemoryLockedClocks(self, minMemClockMHz: int, maxMemClockMHz: int) -> None:
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        with contextlib.suppress(pynvml.NVMLError_NotSupported):
            pynvml.nvmlDeviceSetMemoryLockedClocks(handle, minMemClockMHz, maxMemClockMHz)
    
    def GetName(self) -> str:
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        return pynvml.nvmlDeviceGetName(handle)
    
    def GetSupportedMemoryClocks() -> list[int]:
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        return pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)
    
    def GetSupportedGraphicsClocks(freq: int) -> list[int]:
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        return pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, freq)



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
    
    def SetPowerManagementLimit(self, value: int, default : bool = False) -> None:
        handle = amdsmi.amdsmi_get_processor_handles()[self.gpu_index]
        """
        Input parameters:
        processor_handle handle for the given device
        sensor_ind a 0-based sensor index. Normally, this will be 0. If a device has more than one sensor, it could be greater than 0
        cap int that indicates the desired power cap, in microwatts
        """
        if default:
            info = amdsmi.amdsmi_get_power_cap_info(handle)
            amdsmi.amdsmi_set_power_cap(handle, sensor_id=0, cap=info.default_power_cap)
        amdsmi.amdsmi_set_power_cap(handle, sensor_id=0, cap=value)
    
    def SetMemoryLockedClocks(self, minMemClockMHz: int, maxMemClockMHz: int) -> None:
        handle = amdsmi.amdsmi_get_processor_handles()[self.gpu_index]
        amdsmi.amdsmi_set_gpu_clk_range(handle, minMemClockMHz, maxMemClockMHz, clk_type= amdsmi.AMDSMI_CLK_TYPE_SYS)

    def GetName(self) -> str:
        handle = amdsmi.amdsmi_get_processor_handles()[self.gpu_index]
        market_name, vendor_id, device_id, rev_id, asic_serial = amdsmi.amdsmi_get_gpu_asic_info(handle) #TODO: Does this return correct string
        return market_name
    
    def GetSupportedMemoryClocks() -> list[int]:
        handle = amdsmi.amdsmi_get_processor_handles()[self.gpu_index]
        num_supported, current, frequency = amdsmi.amdsmi_get_clk_freq(handle, clk_type=amdsmi.AMDSMI_CLK_TYPE_SYS) #TODO: Figure out correct clk_type
        """ frequency; List of frequencies, only the first num_supported frequencies are valid"""
        return frequency[:num_supported]

    def GetSupportedGraphicsClocks(freq: int) -> list[int]:
        # TODO: what does 137-140 in optimizer.py do?
        handle = amdsmi.amdsmi_get_processor_handles()[self.gpu_index]
        

class RemoteAMDGPU(NativeAMDGPU):
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
    
    def SetPowerManagementLimit(self, index: int, value: int, default: bool = False) -> None:
        self.gpus[index].SetPowerManagementLimit(value, default)
    
    def SetMemoryLockedClocks(self, index: int, minMemClockMHz: int, maxMemClockMHz: int) -> None:
        self.gpus[index].SetMemoryLockedClocks(minMemClockMHz, maxMemClockMHz)

    def GetSupportedMemoryClocks(self, index: int) -> list[int]:
        self.gpus[index].GetSupportedMemoryClocks()

    def GetSupportedGraphicsClocks(self, index: int, freq: int) -> list[int]:
        self.gpus[index].GetSupportedGraphicsClocks(freq)
    
    def GetName(self, index: int) -> str:
        return self.gpus[index].GetName()


"""

should NVIDIAGPUs track all GPUs or only the ones visible to CUDA? I don't see the need
to track all GPUs, but maybe there are some edge cases where it's useful.

should AMD GPUs look at CUDA_VISIBLE_DEVICES? no just ROCM_VISIBLE_DEVICES right?

in the code there is a lot of self.gpu_handles. Should be removed and abstracted away?

why is nvmlDeviceSetPersistenceMode necessary?

no throw destructor?

different subclasses for different types of GPUs? ex. NVIDIA volta vs ampere?

what does 137-140 in optimizer.py do?

GPU manager classes have the same impl, except for the init and shutdown methods.

why is with contextlib.suppress(pynvml.NVMLError_NotSupported):
            pynvml.nvmlDeviceSetMemoryLockedClocks(handle, minMemClockMHz, maxMemClockMHz)
the with needed? something similar needed for amd?
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