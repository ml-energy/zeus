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

import abc

class GPU(abc.ABC):
    def __init__(self, gpu_index: int) -> None:
        self.gpu_index = gpu_index

    @abc.abstractmethod
    def get_total_energy_consumption(self) -> float:
        pass

    @abc.abstractmethod
    def set_power_limit(self, value: int) -> None:
        pass

class NVIDIALocalGPU(GPU):
    def __init__(self, gpu_index: int) -> None:
        super().__init__(gpu_index)

    def get_total_energy_consumption(self) -> float:
        # Call NVML
        pass

    def set_power_limit(self, value: int) -> None:
        # Call NVML
        pass


class NVIDIARemoteGPU(GPU):
    def __init__(self, gpu_index: int, server_address: str) -> None:
        super().__init__(gpu_index)
        self.server_address = server_address

    def set_power_limit(self, value: int) -> None:
        # Call server since SYS_ADMIN is required
        pass


# Factory function to get GPU
def get_gpu(gpu_index: int, server_address: str | None = None) -> GPU:
    if server_address is None:
        return NVIDIALocalGPU(gpu_index)
    else:
        return NVIDIARemoteGPU(gpu_index, server_address)