"""GPU device module for Zeus. Abstraction of GPU devices.

The main function of this module is [`get_gpus`][zeus.device.gpu.get_gpus], which returns a GPU Manager object specific to the platform.
To instantiate a GPU Manager object, you can do the following:
    
```python
from zeus.device import get_gpus
gpus = get_gpus() # Returns NVIDIAGPUs() or AMDGPUs() depending on the platform.
```

There exists a 1:1 mapping between specific library functions and methods implemented in the GPU Manager object.
For example, for NVIDIA systems, if you wanted to do:

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
constraints = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
```

You can now do:

```python
gpus = get_gpus() # returns a NVIDIAGPUs object
constraints =  gpus.getPowerManagementLimitConstraints(gpu_index)
```

Class hierarchy:

- [`GPUs`][zeus.device.gpu.GPUs]: Abstract class for GPU managers.
    - [`NVIDIAGPUs`][zeus.device.gpu.NVIDIAGPUs]: GPU manager for NVIDIA GPUs, initialize NVIDIAGPU objects.
    - [`AMDGPUs`][zeus.device.gpu.AMDGPUs]: GPU manager for AMD GPUs, initialize AMDGPU objects.
- [`GPU`][zeus.device.gpu.GPU]: Abstract class for GPU objects.
    - [`NVIDIAGPU`][zeus.device.gpu.NVIDIAGPU]: GPU object for NVIDIA GPUs.
    - [`AMDGPU`][zeus.device.gpu.AMDGPU]: GPU object for AMD GPUs.


The following exceptions are defined in this module:

- [`ZeusGPUInitError`][zeus.device.gpu.ZeusGPUInitError]: Base class for initialization errors.
- [`ZeusGPUInvalidArgError`][zeus.device.gpu.ZeusGPUInvalidArgError]: Error for invalid arguments.
- [`ZeusGPUNotSupportedError`][zeus.device.gpu.ZeusGPUNotSupportedError]: Error for unsupported GPUs.
- [`ZeusGPUNoPermissionError`][zeus.device.gpu.ZeusGPUNoPermissionError]: Error for permission issues.
- [`ZeusGPUAlreadyInitializedError`][zeus.device.gpu.ZeusGPUAlreadyInitializedError]: Error for reinitialization.
- [`ZeusGPUNotFoundError`][zeus.device.gpu.ZeusGPUNotFoundError]: Error for missing GPUs.
- [`ZeusGPUInsufficientSizeError`][zeus.device.gpu.ZeusGPUInsufficientSizeError]: Error for insufficient buffer size.
- [`ZeusGPUInsufficientPowerError`][zeus.device.gpu.ZeusGPUInsufficientPowerError]: Error for insufficient power.
- [`ZeusGPUDriverNotLoadedError`][zeus.device.gpu.ZeusGPUDriverNotLoadedError]: Error for driver issues.
- [`ZeusGPUTimeoutError`][zeus.device.gpu.ZeusGPUTimeoutError]: Error for timeout issues.
- [`ZeusGPUIRQError`][zeus.device.gpu.ZeusGPUIRQError]: Error for IRQ issues.
- [`ZeusGPULibraryNotFoundError`][zeus.device.gpu.ZeusGPULibraryNotFoundError]: Error for missing libraries.
- [`ZeusGPUFunctionNotFoundError`][zeus.device.gpu.ZeusGPUFunctionNotFoundError]: Error for missing functions.
- [`ZeusGPUCorruptedInfoROMError`][zeus.device.gpu.ZeusGPUCorruptedInfoROMError]: Error for corrupted info ROM.
- [`ZeusGPULostError`][zeus.device.gpu.ZeusGPULostError]: Error for lost GPUs.
- [`ZeusGPUResetRequiredError`][zeus.device.gpu.ZeusGPUResetRequiredError]: Error for GPUs requiring reset.
- [`ZeusGPUOperatingSystemError`][zeus.device.gpu.ZeusGPUOperatingSystemError]: Error for OS issues.
- [`ZeusGPULibRMVersionMismatchError`][zeus.device.gpu.ZeusGPULibRMVersionMismatchError]: Error for library version mismatch.
- [`ZeusGPUMemoryError`][zeus.device.gpu.ZeusGPUMemoryError]: Error for memory issues.
- [`ZeusGPUUnknownError`][zeus.device.gpu.ZeusGPUUnknownError]: Error for unknown issues.


"""
from zeus.device.gpu import (
    get_gpus,
    ZeusGPUInitError,
    ZeusGPUInvalidArgError,
    ZeusGPUNotSupportedError,
    ZeusGPUNoPermissionError,
    ZeusGPUAlreadyInitializedError,
    ZeusGPUNotFoundError,
    ZeusGPUInsufficientSizeError,
    ZeusGPUInsufficientPowerError,
    ZeusGPUDriverNotLoadedError,
    ZeusGPUTimeoutError,
    ZeusGPUIRQError,
    ZeusGPULibraryNotFoundError,
    ZeusGPUFunctionNotFoundError,
    ZeusGPUCorruptedInfoROMError,
    ZeusGPULostError,
    ZeusGPUResetRequiredError,
    ZeusGPUOperatingSystemError,
    ZeusGPULibRMVersionMismatchError,
    ZeusGPUMemoryError,
    ZeusGPUUnknownError,
)
