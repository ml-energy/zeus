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
from typing import TYPE_CHECKING

import abc
import pynvml
import os

# from exception import ZeusBaseGPUError
if TYPE_CHECKING:
    import pynvml


# import amdsmi

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


#unpriv nvidia privel
class UnprivilegedNVIDIAGPU(NVIDIAGPU):
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


"""

should NVIDIAGPUs track all GPUs or only the ones visible to CUDA? I don't see the need
to track all GPUs, but maybe there are some edge cases where it's useful.

in the code there is a lot of self.gpu_handles. Should be removed and abstracted away?


no throw destructor?

different subclasses for different types of GPUs? ex. NVIDIA volta vs ampere?

what does 137-140 in optimizer.py do?

GPU manager classes have the same impl, except for the init and shutdown methods.

why is with contextlib.suppress(pynvml.NVMLError_NotSupported):
            pynvml.nvmlDeviceSetMemoryLockedClocks(handle, minMemClockMHz, maxMemClockMHz)
the with needed? something similar needed for amd?



style for methods? snake case

__len__ -> returns number of gpus
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
        self.gpus = [NVIDIAGPU(index) for index in self.visible_indices]

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
        self.gpus = [AMDGPU(index) for index in self.visible_indices]

    def __del__(self) -> None:
        amdsmi.amdsmi_shut_down()



# Ensure only one instance of GPUs is created
def get_gpus() -> GPUs:
    return NVIDIAGPUs()

gpus = get_gpus()



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