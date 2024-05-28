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
from __future__ import annotations

from zeus.device.gpu.common import *
from zeus.device.gpu.common import GPUs, ZeusGPUInitError
from zeus.device.gpu.nvidia import nvml_is_available, NVIDIAGPUs
from zeus.device.gpu.amd import amdsmi_is_available, AMDGPUs


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
