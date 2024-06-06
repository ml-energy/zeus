"""Abstraction layer for GPU devices.

The main function of this module is [`get_gpus`][zeus.device.gpu.get_gpus],
which returns a GPU Manager object specific to the platform.

!!! Important
    In theory, any NVIDIA GPU would be supported.
    On the other hand, for AMD GPUs, we currently only support ROCm 6.1 and later.

## Getting handles to GPUs

The main API exported from this module is the `get_gpus` function. It returns either
[`NVIDIAGPUs`][zeus.device.gpu.nvidia.NVIDIAGPUs] or [`AMDGPUs`][zeus.device.gpu.amd.AMDGPUs]
depending on the platform. 

```python
from zeus.device import get_gpus
gpus = get_gpus()  
```

## Calling GPU management APIs

GPU management library APIs are mapped to methods on [`GPU`][zeus.device.gpu.common.GPU].

For example, for NVIDIA GPUs (which uses `pynvml`), you would have called:

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
constraints = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
```

With the Zeus GPU abstraction layer, you would now call:

```python
gpus = get_gpus() # returns an NVIDIAGPUs object
constraints = gpus.getPowerManagementLimitConstraints(gpu_index)
```

## Non-blocking calls

Some implementations of `GPU` support non-blocking calls to setters.
If non-blocking calls are not supported, setting `block` will be ignored and the call will block.
Check [`GPU.supports_non_blocking`][zeus.device.gpu.common.GPU.supports_nonblocking_setters]
to see if non-blocking calls are supported.
Note that non-blocking calls will not raise exceptions even if the call fails.

Currently, only [`ZeusdNVIDIAGPU`][zeus.device.gpu.nvidia.ZeusdNVIDIAGPU] supports non-blocking calls
to methods that set the GPU's power limit, GPU frequency, memory frequency, and persistence mode.
This is possible because the Zeus daemon supports a `block: bool` parameter in HTTP requests,
which can be set to `False` to make the call return immediately without checking the result.

## Error handling

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
    """Initialize and return a singleton object for GPU management.

    This function returns a GPU management object that aims to abstract
    the underlying GPU vendor and their specific monitoring library
    (pynvml for NVIDIA GPUs and amdsmi for AMD GPUs). Management APIs
    are mapped to methods on the returned [`GPUs`][zeus.device.gpu.GPUs] object.

    GPU availability is checked in the following order:

    1. NVIDIA GPUs using `pynvml`
    1. AMD GPUs using `amdsmi`
    1. If both are unavailable, a `ZeusGPUInitError` is raised.

    Args:
        ensure_homogeneous (bool): If True, ensures that all tracked GPUs have the same name.
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
