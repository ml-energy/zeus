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


from amdsmi import amdsmi_init

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
    pass

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


# ?: Enfore singleton pattern?

# NVIDIA GPUs
class NVIDIAGPUs(GPUs):
    def __init__(self, ensure_homogeneuous: bool = True) -> None:
        pynvml.nvmlInit()

    def __del__(self) -> None:
        pynvml.nvmlShutdown()


#AMD GPUs

class AMDGPUs(GPUs):
    def __init__(self, ensure_homogeneuous: bool = True) -> None:
        amdsmi_init()

    def Init(self) -> None:
        pass

    def Shutdown(self) -> None:
        pass

    def get_gpu_count(self) -> int:
        pass

    def get_total_energy_consumption(self) -> float:
        pass

    def set_power_limit(self, value: int) -> None:
        pass


# Ensure only one instance of GPUs is created
def get_gpus() -> GPUs:
    return NVIDIAGPUs() if pynvml.is_initialized() else AMDGPUs()

GPUManager = get_gpus()

