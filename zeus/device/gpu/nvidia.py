"""NVIDIA GPUs."""

from __future__ import annotations

import os
import warnings
import functools
import contextlib
from pathlib import Path
from typing import Sequence
from functools import lru_cache

import httpx
import pynvml

import zeus.device.gpu.common as gpu_common
from zeus.device.exception import ZeusdError
from zeus.utils.logging import get_logger

logger = get_logger(name=__name__)


@lru_cache(maxsize=1)
def nvml_is_available() -> bool:
    """Check if NVML is available."""
    try:
        import pynvml
    except ImportError:
        logger.info(
            "Failed to import `pynvml`. Make sure you have `nvidia-ml-py` installed."
        )
        return False

    # Detect unofficial pynvml packages.
    # If detected, this should be a critical error.
    if not hasattr(pynvml, "_nvmlGetFunctionPointer"):
        logger.error("Unoffical pynvml package detected!")
        raise ImportError(
            "Unofficial pynvml package detected! "
            "This causes conflicts with the official NVIDIA bindings. "
            "Please remove with `pip uninstall pynvml` and instead use the official "
            "bindings from NVIDIA: `nvidia-ml-py`. "
        )

    try:
        pynvml.nvmlInit()
        logger.info("pynvml is available and initialized.")
        return True
    except pynvml.NVMLError as e:
        logger.info("pynvml is available but could not initialize NVML: %s.", e)
        return False


def _handle_nvml_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except pynvml.NVMLError as e:
            exception_class = NVIDIAGPU._exception_map.get(
                e.value,  # pyright: ignore[reportAttributeAccessIssue]
                gpu_common.ZeusGPUUnknownError,
            )
            raise exception_class(str(e)) from e

    return wrapper


class NVIDIAGPU(gpu_common.GPU):
    """Implementation of `GPU` for NVIDIA GPUs."""

    def __init__(self, gpu_index: int) -> None:
        """Initialize the GPU object."""
        super().__init__(gpu_index)
        self._get_handle()
        self._supportsGetTotalEnergyConsumption = None

    _exception_map = {
        pynvml.NVML_ERROR_UNINITIALIZED: gpu_common.ZeusGPUInitError,
        pynvml.NVML_ERROR_INVALID_ARGUMENT: gpu_common.ZeusGPUInvalidArgError,
        pynvml.NVML_ERROR_NOT_SUPPORTED: gpu_common.ZeusGPUNotSupportedError,
        pynvml.NVML_ERROR_NO_PERMISSION: gpu_common.ZeusGPUNoPermissionError,
        pynvml.NVML_ERROR_ALREADY_INITIALIZED: gpu_common.ZeusGPUAlreadyInitializedError,
        pynvml.NVML_ERROR_NOT_FOUND: gpu_common.ZeusGPUNotFoundError,
        pynvml.NVML_ERROR_INSUFFICIENT_SIZE: gpu_common.ZeusGPUInsufficientSizeError,
        pynvml.NVML_ERROR_INSUFFICIENT_POWER: gpu_common.ZeusGPUInsufficientPowerError,
        pynvml.NVML_ERROR_DRIVER_NOT_LOADED: gpu_common.ZeusGPUDriverNotLoadedError,
        pynvml.NVML_ERROR_TIMEOUT: gpu_common.ZeusGPUTimeoutError,
        pynvml.NVML_ERROR_IRQ_ISSUE: gpu_common.ZeusGPUIRQError,
        pynvml.NVML_ERROR_LIBRARY_NOT_FOUND: gpu_common.ZeusGPULibraryNotFoundError,
        pynvml.NVML_ERROR_FUNCTION_NOT_FOUND: gpu_common.ZeusGPUFunctionNotFoundError,
        pynvml.NVML_ERROR_CORRUPTED_INFOROM: gpu_common.ZeusGPUCorruptedInfoROMError,
        pynvml.NVML_ERROR_GPU_IS_LOST: gpu_common.ZeusGPULostError,
        pynvml.NVML_ERROR_RESET_REQUIRED: gpu_common.ZeusGPUResetRequiredError,
        pynvml.NVML_ERROR_OPERATING_SYSTEM: gpu_common.ZeusGPUOperatingSystemError,
        pynvml.NVML_ERROR_LIB_RM_VERSION_MISMATCH: gpu_common.ZeusGPULibRMVersionMismatchError,
        pynvml.NVML_ERROR_MEMORY: gpu_common.ZeusGPUMemoryError,
        pynvml.NVML_ERROR_UNKNOWN: gpu_common.ZeusGPUUnknownError,
    }

    @_handle_nvml_errors
    def _get_handle(self):
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

    @_handle_nvml_errors
    def getName(self) -> str:
        """Return the name of the GPU model."""
        return pynvml.nvmlDeviceGetName(self.handle)

    @property
    def supports_nonblocking_setters(self) -> bool:
        """Return True if the GPU object supports non-blocking configuration setters."""
        return False

    @_handle_nvml_errors
    def getPowerManagementLimitConstraints(self) -> tuple[int, int]:
        """Return the minimum and maximum power management limits. Units: mW."""
        min_, max_ = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(self.handle)
        return (min_, max_)

    @_handle_nvml_errors
    def setPowerManagementLimit(self, power_limit_mw: int, _block: bool = True) -> None:
        """Set the GPU's power management limit. Unit: mW."""
        pynvml.nvmlDeviceSetPowerManagementLimit(self.handle, power_limit_mw)

    @_handle_nvml_errors
    def resetPowerManagementLimit(self, _block: bool = True) -> None:
        """Reset the GPU's power management limit to the default value."""
        pynvml.nvmlDeviceSetPowerManagementLimit(
            self.handle,
            pynvml.nvmlDeviceGetPowerManagementDefaultLimit(self.handle),
        )

    @_handle_nvml_errors
    def setPersistenceMode(self, enabled: bool, _block: bool = True) -> None:
        """Set persistence mode."""
        if enabled:
            pynvml.nvmlDeviceSetPersistenceMode(
                self.handle, pynvml.NVML_FEATURE_ENABLED
            )
        else:
            pynvml.nvmlDeviceSetPersistenceMode(
                self.handle, pynvml.NVML_FEATURE_DISABLED
            )

    @_handle_nvml_errors
    def getSupportedMemoryClocks(self) -> list[int]:
        """Return a list of supported memory clock frequencies. Units: MHz."""
        return pynvml.nvmlDeviceGetSupportedMemoryClocks(self.handle)

    @_handle_nvml_errors
    def setMemoryLockedClocks(
        self, min_clock_mhz: int, max_clock_mhz: int, _block: bool = True
    ) -> None:
        """Lock the memory clock to a specified range. Units: MHz."""
        pynvml.nvmlDeviceSetMemoryLockedClocks(
            self.handle, min_clock_mhz, max_clock_mhz
        )

    @_handle_nvml_errors
    def resetMemoryLockedClocks(self, _block: bool = True) -> None:
        """Reset the locked memory clocks to the default."""
        pynvml.nvmlDeviceResetMemoryLockedClocks(self.handle)

    @_handle_nvml_errors
    def getSupportedGraphicsClocks(
        self, memory_clock_mhz: int | None = None
    ) -> list[int]:
        """Return a list of supported graphics clock frequencies. Units: MHz.

        Args:
            memory_clock_mhz: Memory clock frequency to use. Some GPUs have
                different supported graphics clocks depending on the memory clock.
        """
        pass
        return pynvml.nvmlDeviceGetSupportedGraphicsClocks(
            self.handle, memory_clock_mhz
        )

    @_handle_nvml_errors
    def setGpuLockedClocks(
        self, min_clock_mhz: int, max_clock_mhz: int, _block: bool = True
    ) -> None:
        """Lock the GPU clock to a specified range. Units: MHz."""
        pynvml.nvmlDeviceSetGpuLockedClocks(self.handle, min_clock_mhz, max_clock_mhz)

    @_handle_nvml_errors
    def resetGpuLockedClocks(self, _block: bool = True) -> None:
        """Reset the locked GPU clocks to the default."""
        pynvml.nvmlDeviceResetGpuLockedClocks(self.handle)

    @_handle_nvml_errors
    def getAveragePowerUsage(self) -> int:
        """Return the average power draw of the GPU. Units: mW."""
        metric = pynvml.nvmlDeviceGetFieldValues(
            self.handle, [pynvml.NVML_FI_DEV_POWER_AVERAGE]
        )[0]
        if (ret := metric.nvmlReturn) != pynvml.NVML_SUCCESS:
            raise pynvml.NVMLError(ret)
        return metric.value.uiVal

    @_handle_nvml_errors
    def getInstantPowerUsage(self) -> int:
        """Return the current power draw of the GPU. Units: mW."""
        metric = pynvml.nvmlDeviceGetFieldValues(
            self.handle, [pynvml.NVML_FI_DEV_POWER_INSTANT]
        )[0]
        if (ret := metric.nvmlReturn) != pynvml.NVML_SUCCESS:
            raise pynvml.NVMLError(ret)
        return metric.value.uiVal

    @_handle_nvml_errors
    def getAverageMemoryPowerUsage(self) -> int:
        """Return the average power draw of the GPU's memory. Units: mW.

        !!! Warning
            This isn't exactly documented in NVML at the time of writing, but `nvidia-smi`
            makes use of this API.

            Confirmed working on H100 80GB HBM3. Confirmed not working on A40.
        """
        metric = pynvml.nvmlDeviceGetFieldValues(
            self.handle,
            [(pynvml.NVML_FI_DEV_POWER_AVERAGE, pynvml.NVML_POWER_SCOPE_MEMORY)],
        )[0]
        if (ret := metric.nvmlReturn) != pynvml.NVML_SUCCESS:
            raise pynvml.NVMLError(ret)
        power = metric.value.uiVal
        if power == 0:
            warnings.warn(
                "Average memory power returned 0. The current GPU may not be supported.",
                stacklevel=1,
            )
        return power

    @_handle_nvml_errors
    def supportsGetTotalEnergyConsumption(self) -> bool:
        """Check if the GPU supports retrieving total energy consumption."""
        # Supported on Volta or newer microarchitectures
        if self._supportsGetTotalEnergyConsumption is None:
            self._supportsGetTotalEnergyConsumption = (
                pynvml.nvmlDeviceGetArchitecture(self.handle)
                >= pynvml.NVML_DEVICE_ARCH_VOLTA
            )

        return self._supportsGetTotalEnergyConsumption

    @_handle_nvml_errors
    def getTotalEnergyConsumption(self) -> int:
        """Return the total energy consumption of the specified GPU. Units: mJ."""
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)


class ZeusdNVIDIAGPU(NVIDIAGPU):
    """An NVIDIAGPU that sets GPU knobs that require `SYS_ADMIN` via zeusd.

    Some NVML APIs (e.g., setting persistence mode, power limit, frequency)
    requires the Linux security capability `SYS_ADMIN`, which is virtually `sudo`.
    This class overrides those methods so that they send a request to the
    Zeus daemon.

    See [here](https://ml.energy/zeus/getting_started/#system-privileges)
    for details on system privileges required.
    """

    def __init__(
        self,
        gpu_index: int,
        zeusd_sock_path: str = "/var/run/zeusd.sock",
    ) -> None:
        """Initialize NVML and sets up the GPUs.

        Args:
            gpu_index (int): Index of the GPU.
            zeusd_sock_path (str): Path to the Zeus daemon socket.
        """
        super().__init__(gpu_index)
        self.zeusd_sock_path = zeusd_sock_path

        self._client = httpx.Client(transport=httpx.HTTPTransport(uds=zeusd_sock_path))
        self._url_prefix = f"http://zeusd/gpu/{gpu_index}"

    @property
    def supports_nonblocking_setters(self) -> bool:
        """Return True if the GPU object supports non-blocking configuration setters."""
        return True

    def setPowerManagementLimit(self, power_limit_mw: int, block: bool = True) -> None:
        """Set the GPU's power management limit. Unit: mW."""
        resp = self._client.post(
            self._url_prefix + "/set_power_limit",
            json=dict(power_limit_mw=power_limit_mw, block=block),
        )
        if resp.status_code != 200:
            raise ZeusdError(f"Failed to set power management limit: {resp.text}")
        logger.debug("Took %s ms to set power limit", resp.elapsed.microseconds / 1000)

    @_handle_nvml_errors
    def resetPowerManagementLimit(self, block: bool = True) -> None:
        """Reset the GPU's power management limit to the default value."""
        self.setPowerManagementLimit(
            pynvml.nvmlDeviceGetPowerManagementDefaultLimit(self.handle),
            block,
        )

    def setPersistenceMode(self, enabled: bool, block: bool = False) -> None:
        """Set persistence mode."""
        resp = self._client.post(
            self._url_prefix + "/set_persistence_mode",
            json=dict(enabled=enabled, block=block),
        )
        if resp.status_code != 200:
            raise ZeusdError(f"Failed to set persistence mode: {resp.text}")
        logger.debug(
            "Took %s ms to set persistence mode", resp.elapsed.microseconds / 1000
        )

    def setMemoryLockedClocks(
        self, min_clock_mhz: int, max_clock_mhz: int, block: bool = True
    ) -> None:
        """Lock the memory clock to a specified range. Units: MHz."""
        resp = self._client.post(
            self._url_prefix + "/set_mem_locked_clocks",
            json=dict(
                min_clock_mhz=min_clock_mhz, max_clock_mhz=max_clock_mhz, block=block
            ),
        )
        if resp.status_code != 200:
            raise ZeusdError(f"Failed to set memory locked clocks: {resp.text}")
        logger.debug(
            "Took %s ms to set memory locked clocks", resp.elapsed.microseconds / 1000
        )

    def resetMemoryLockedClocks(self, block: bool = True) -> None:
        """Reset the locked memory clocks to the default."""
        resp = self._client.post(
            self._url_prefix + "/reset_mem_locked_clocks", json=dict(block=block)
        )
        if resp.status_code != 200:
            raise ZeusdError(f"Failed to reset memory locked clocks: {resp.text}")

    def setGpuLockedClocks(
        self, min_clock_mhz: int, max_clock_mhz: int, block: bool = True
    ) -> None:
        """Lock the GPU clock to a specified range. Units: MHz."""
        resp = self._client.post(
            self._url_prefix + "/set_gpu_locked_clocks",
            json=dict(
                min_clock_mhz=min_clock_mhz, max_clock_mhz=max_clock_mhz, block=block
            ),
        )
        if resp.status_code != 200:
            raise ZeusdError(f"Failed to set GPU locked clocks: {resp.text}")

    def resetGpuLockedClocks(self, block: bool = True) -> None:
        """Reset the locked GPU clocks to the default."""
        resp = self._client.post(
            self._url_prefix + "/reset_gpu_locked_clocks", json=dict(block=block)
        )
        if resp.status_code != 200:
            raise ZeusdError(f"Failed to reset GPU locked clocks: {resp.text}")


class NVIDIAGPUs(gpu_common.GPUs):
    """Implementation of `GPUs` for NVIDIA GPUs.

    `CUDA_VISIBLE_DEVICES` environment variable is respected if set.
    For example, if there are 4 GPUs on the node and `CUDA_VISIBLE_DEVICES=0,2`,
    only GPUs 0 and 2 are instantiated. In this case, to access
    GPU of CUDA index 0, use the index 0, and for CUDA index 2, use the index 1.

    If you have the Zeus daemon deployed, make sure you have set the `ZEUSD_SOCK_PATH`
    environment variable to the path of the Zeus daemon socket. This class will
    automatically use [`ZeusdNVIDIAGPU`][zeus.device.gpu.nvidia.ZeusdNVIDIAGPU]
    if `ZEUSD_SOCK_PATH` is set.
    """

    def __init__(self, ensure_homogeneous: bool = False) -> None:
        """Initialize NVML and sets up the GPUs.

        Args:
            ensure_homogeneous (bool): If True, ensures that all tracked GPUs have the same name.
        """
        try:
            pynvml.nvmlInit()
            self._init_gpus()
            if ensure_homogeneous:
                self._ensure_homogeneous()
        except pynvml.NVMLError as e:
            exception_class = NVIDIAGPU._exception_map.get(
                e.value,  # pyright: ignore[reportAttributeAccessIssue]
                gpu_common.ZeusBaseGPUError,
            )
            raise exception_class(
                e.msg  # pyright: ignore[reportAttributeAccessIssue]
            ) from e

    @property
    def gpus(self) -> Sequence[NVIDIAGPU]:
        """Return a list of NVIDIAGPU objects being tracked."""
        return self._gpus

    def _init_gpus(self) -> None:
        # Must respect `CUDA_VISIBLE_DEVICES` if set
        if (visible_device := os.environ.get("CUDA_VISIBLE_DEVICES")) is not None:
            if not visible_device:
                raise gpu_common.ZeusGPUInitError(
                    "CUDA_VISIBLE_DEVICES is set to an empty string. "
                    "It should either be unset or a comma-separated list of GPU indices."
                )
            if visible_device.startswith("MIG"):
                raise gpu_common.ZeusGPUInitError(
                    "CUDA_VISIBLE_DEVICES contains MIG devices. NVML (the library used by Zeus) "
                    "currently does not support measuring the power or energy consumption of MIG "
                    "slices. You can still measure the whole GPU by temporarily setting "
                    "CUDA_VISIBLE_DEVICES to integer GPU indices and restoring it afterwards."
                )
            visible_indices = [int(idx) for idx in visible_device.split(",")]
        else:
            visible_indices = list(range(pynvml.nvmlDeviceGetCount()))

        # If `ZEUSD_SOCK_PATH` is set, always use ZeusdNVIDIAGPU
        if (sock_path := os.environ.get("ZEUSD_SOCK_PATH")) is not None:
            if not Path(sock_path).exists():
                raise ZeusdError(
                    f"ZEUSD_SOCK_PATH points to non-existent file: {sock_path}"
                )
            if not Path(sock_path).is_socket():
                raise ZeusdError(f"ZEUSD_SOCK_PATH is not a socket: {sock_path}")
            if not os.access(sock_path, os.W_OK):
                raise ZeusdError(f"ZEUSD_SOCK_PATH is not writable: {sock_path}")
            self._gpus = [
                ZeusdNVIDIAGPU(gpu_num, sock_path) for gpu_num in visible_indices
            ]
            # Disable the warning about SYS_ADMIN capabilities
            self._disable_sys_admin_warning = True

        # Otherwise just use NVIDIAGPU
        else:
            self._gpus = [NVIDIAGPU(gpu_num) for gpu_num in visible_indices]

    def __del__(self) -> None:
        """Shut down NVML."""
        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlShutdown()
