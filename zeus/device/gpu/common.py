"""Error wrappers and classes common to all GPU vendors."""

from __future__ import annotations

import abc
import logging
import warnings
from typing import Sequence

from zeus.device.exception import ZeusBaseGPUError
from zeus.device.common import has_sys_admin, deprecated_alias, DeprecatedAliasABCMeta

logger = logging.getLogger(__name__)


class GPU(abc.ABC, metaclass=DeprecatedAliasABCMeta):
    """Abstract base class for managing one GPU.

    For each method, child classes should call into vendor-specific
    GPU management libraries (e.g., NVML for NVIDIA GPUs).
    """

    def __init__(self, gpu_index: int) -> None:
        """Initializ the GPU with a specified index."""
        self.gpu_index = gpu_index

    @property
    @abc.abstractmethod
    def supports_nonblocking_setters(self) -> bool:
        """Return True if the GPU object supports non-blocking configuration setters."""
        return False

    @deprecated_alias("getName")
    @abc.abstractmethod
    def get_name(self) -> str:
        """Return the name of the GPU model."""
        pass

    @deprecated_alias("getPowerManagementLimitConstraints")
    @abc.abstractmethod
    def get_power_management_limit_constraints(self) -> tuple[int, int]:
        """Return the minimum and maximum power management limits. Units: mW."""
        pass

    @abc.abstractmethod
    def get_power_management_limit(self) -> int:
        """Return the current power management limit. Units: mW."""
        pass

    @deprecated_alias("setPowerManagementLimit")
    @abc.abstractmethod
    def set_power_management_limit(self, power_limit_mw: int, block: bool = True) -> None:
        """Set the GPU's power management limit. Unit: mW."""
        pass

    @deprecated_alias("resetPowerManagementLimit")
    @abc.abstractmethod
    def reset_power_management_limit(self, block: bool = True) -> None:
        """Reset the GPU's power management limit to the default value."""
        pass

    @deprecated_alias("setPersistenceMode")
    @abc.abstractmethod
    def set_persistence_mode(self, enabled: bool, block: bool = True) -> None:
        """Set persistence mode."""
        pass

    @deprecated_alias("getSupportedMemoryClocks")
    @abc.abstractmethod
    def get_supported_memory_clocks(self) -> list[int]:
        """Return a list of supported memory clock frequencies. Units: MHz."""
        pass

    @deprecated_alias("setMemoryLockedClocks")
    @abc.abstractmethod
    def set_memory_locked_clocks(self, min_clock_mhz: int, max_clock_mhz: int, block: bool = True) -> None:
        """Lock the memory clock to a specified range. Units: MHz."""
        pass

    @deprecated_alias("resetMemoryLockedClocks")
    @abc.abstractmethod
    def reset_memory_locked_clocks(self, block: bool = True) -> None:
        """Reset the locked memory clocks to the default."""
        pass

    @deprecated_alias("getSupportedGraphicsClocks")
    @abc.abstractmethod
    def get_supported_graphics_clocks(self, memory_clock_mhz: int | None = None) -> list[int]:
        """Return a list of supported graphics clock frequencies. Units: MHz.

        Args:
            memory_clock_mhz: Memory clock frequency to use. Some GPUs have
                different supported graphics clocks depending on the memory clock.
        """
        pass

    @deprecated_alias("setGpuLockedClocks")
    @abc.abstractmethod
    def set_gpu_locked_clocks(self, min_clock_mhz: int, max_clock_mhz: int, block: bool = True) -> None:
        """Lock the GPU clock to a specified range. Units: MHz."""
        pass

    @deprecated_alias("resetGpuLockedClocks")
    @abc.abstractmethod
    def reset_gpu_locked_clocks(self, block: bool = True) -> None:
        """Reset the locked GPU clocks to the default."""
        pass

    @deprecated_alias("getAveragePowerUsage")
    @abc.abstractmethod
    def get_average_power_usage(self) -> int:
        """Return the average power usage of the GPU. Units: mW."""
        pass

    @deprecated_alias("getInstantPowerUsage")
    @abc.abstractmethod
    def get_instant_power_usage(self) -> int:
        """Return the current power draw of the GPU. Units: mW."""
        pass

    @deprecated_alias("getAverageMemoryPowerUsage")
    @abc.abstractmethod
    def get_average_memory_power_usage(self) -> int:
        """Return the average power usage of the GPU's memory. Units: mW."""
        pass

    @deprecated_alias("supportsGetTotalEnergyConsumption")
    @abc.abstractmethod
    def supports_get_total_energy_consumption(self) -> bool:
        """Check if the GPU supports retrieving total energy consumption."""
        pass

    @deprecated_alias("getTotalEnergyConsumption")
    @abc.abstractmethod
    def get_total_energy_consumption(self) -> int:
        """Return the total energy consumption of the GPU since driver load. Units: mJ."""
        pass

    @deprecated_alias("getGpuTemperature")
    @abc.abstractmethod
    def get_gpu_temperature(self) -> int:
        """Return the current GPU temperature. Units: Celsius."""
        pass


class GPUs(abc.ABC, metaclass=DeprecatedAliasABCMeta):
    """An abstract base class for a collection of `GPU` objects.

    This is basically a list of [`GPU`][zeus.device.gpu.common.GPU] objects and forwards
    most API calls to the individual `GPU` objects. Still, a separate wrapper class is
    is needed to for group-level operations like:

    - `ensure_homogeneous` that ensures that all GPUs have the same name
    - handling vendor-specific environment variables (e.g., `CUDA_VISIBLE_DEVICES` for NVIDIA)
    """

    @abc.abstractmethod
    def __init__(self, ensure_homogeneous: bool = False) -> None:
        """Initialize the GPU management library and initializes `GPU` objects."""
        pass

    @abc.abstractmethod
    def __del__(self) -> None:
        """Shut down the GPU monitoring library to release resources and clean up."""
        pass

    @property
    @abc.abstractmethod
    def gpus(self) -> Sequence[GPU]:
        """Return a list of GPU objects being tracked."""
        pass

    def __len__(self) -> int:
        """Return the number of GPUs being tracked."""
        return len(self.gpus)

    def _ensure_homogeneous(self) -> None:
        """Ensures that all tracked GPUs are homogeneous in terms of name."""
        gpu_names = [gpu.get_name() for gpu in self.gpus]
        # Both zero (no GPUs found) and one are fine.
        if len(set(gpu_names)) > 1:
            raise ZeusGPUHeterogeneousError(f"Heterogeneous GPUs found: {gpu_names}")

    def _warn_sys_admin(self) -> None:
        """Warn the user if the current process doesn't have `SYS_ADMIN` privileges."""
        # Deriving classes can disable this warning by setting this attribute.
        if not getattr(self, "_disable_sys_admin_warning", False) and not has_sys_admin():
            warnings.warn(
                "You are about to call a GPU management API that requires "
                "`SYS_ADMIN` privileges. Some energy optimizers that change the "
                "GPU's power settings need this.\nSee "
                "https://ml.energy/zeus/getting_started/#system-privileges "
                "for more information and how to obtain `SYS_ADMIN`.",
                stacklevel=2,
            )
            # Only warn once.
            self._disable_sys_admin_warning = True

    @deprecated_alias("getName")
    def get_name(self, gpu_index: int) -> str:
        """Return the name of the specified GPU."""
        return self.gpus[gpu_index].get_name()

    @deprecated_alias("getPowerManagementLimitConstraints")
    def get_power_management_limit_constraints(self, gpu_index: int) -> tuple[int, int]:
        """Return the minimum and maximum power management limits. Units: mW."""
        return self.gpus[gpu_index].get_power_management_limit_constraints()

    def get_power_management_limit(self, gpu_index: int) -> int:
        """Return the current power management limit. Units: mW."""
        return self.gpus[gpu_index].get_power_management_limit()

    @deprecated_alias("setPowerManagementLimit")
    def set_power_management_limit(self, gpu_index: int, power_limit_mw: int, block: bool = True) -> None:
        """Set the GPU's power management limit. Unit: mW."""
        self._warn_sys_admin()
        self.gpus[gpu_index].set_power_management_limit(power_limit_mw, block)

    @deprecated_alias("resetPowerManagementLimit")
    def reset_power_management_limit(self, gpu_index: int, block: bool = True) -> None:
        """Reset the GPU's power management limit to the default value."""
        self._warn_sys_admin()
        self.gpus[gpu_index].reset_power_management_limit(block)

    @deprecated_alias("setPersistenceMode")
    def set_persistence_mode(self, gpu_index: int, enabled: bool, block: bool = True) -> None:
        """Set persistence mode for the specified GPU."""
        self._warn_sys_admin()
        self.gpus[gpu_index].set_persistence_mode(enabled, block)

    @deprecated_alias("getSupportedMemoryClocks")
    def get_supported_memory_clocks(self, gpu_index: int) -> list[int]:
        """Return a list of supported memory clock frequencies. Units: MHz."""
        return self.gpus[gpu_index].get_supported_memory_clocks()

    @deprecated_alias("setMemoryLockedClocks")
    def set_memory_locked_clocks(
        self,
        gpu_index: int,
        min_clock_mhz: int,
        max_clock_mhz: int,
        block: bool = True,
    ) -> None:
        """Lock the memory clock to a specified range. Units: MHz."""
        self._warn_sys_admin()
        self.gpus[gpu_index].set_memory_locked_clocks(min_clock_mhz, max_clock_mhz, block)

    @deprecated_alias("resetMemoryLockedClocks")
    def reset_memory_locked_clocks(self, gpu_index: int, block: bool = True) -> None:
        """Reset the locked memory clocks to the default."""
        self._warn_sys_admin()
        self.gpus[gpu_index].reset_memory_locked_clocks(block)

    @deprecated_alias("getSupportedGraphicsClocks")
    def get_supported_graphics_clocks(self, gpu_index: int, memory_clock_mhz: int | None = None) -> list[int]:
        """Return a list of supported graphics clock frequencies. Units: MHz.

        Args:
            gpu_index: Index of the GPU to query.
            memory_clock_mhz: Memory clock frequency to use. Some GPUs have
                different supported graphics clocks depending on the memory clock.
        """
        return self.gpus[gpu_index].get_supported_graphics_clocks(memory_clock_mhz)

    @deprecated_alias("setGpuLockedClocks")
    def set_gpu_locked_clocks(
        self,
        gpu_index: int,
        min_clock_mhz: int,
        max_clock_mhz: int,
        block: bool = True,
    ) -> None:
        """Lock the GPU clock to a specified range. Units: MHz."""
        self._warn_sys_admin()
        self.gpus[gpu_index].set_gpu_locked_clocks(min_clock_mhz, max_clock_mhz, block)

    @deprecated_alias("resetGpuLockedClocks")
    def reset_gpu_locked_clocks(self, gpu_index: int, block: bool = True) -> None:
        """Reset the locked GPU clocks to the default."""
        self._warn_sys_admin()
        self.gpus[gpu_index].reset_gpu_locked_clocks(block)

    @deprecated_alias("getAveragePowerUsage")
    def get_average_power_usage(self, gpu_index: int) -> int:
        """Return the average power usage of the GPU. Units: mW."""
        return self.gpus[gpu_index].get_average_power_usage()

    @deprecated_alias("getInstantPowerUsage")
    def get_instant_power_usage(self, gpu_index: int) -> int:
        """Return the current power draw of the GPU. Units: mW."""
        return self.gpus[gpu_index].get_instant_power_usage()

    @deprecated_alias("getAverageMemoryPowerUsage")
    def get_average_memory_power_usage(self, gpu_index: int) -> int:
        """Return the average power usage of the GPU's memory. Units: mW."""
        return self.gpus[gpu_index].get_average_memory_power_usage()

    @deprecated_alias("supportsGetTotalEnergyConsumption")
    def supports_get_total_energy_consumption(self, gpu_index: int) -> bool:
        """Check if the GPU supports retrieving total energy consumption."""
        return self.gpus[gpu_index].supports_get_total_energy_consumption()

    @deprecated_alias("getTotalEnergyConsumption")
    def get_total_energy_consumption(self, gpu_index: int) -> int:
        """Return the total energy consumption of the GPU since driver load. Units: mJ."""
        return self.gpus[gpu_index].get_total_energy_consumption()

    @deprecated_alias("getGpuTemperature")
    def get_gpu_temperature(self, gpu_index: int) -> int:
        """Return the current GPU temperature. Units: Celsius."""
        return self.gpus[gpu_index].get_gpu_temperature()


class EmptyGPUs(GPUs):
    """A concrete class implementing the GPUs abstract base class, but representing an empty collection of GPUs.

    This class is used to represent a scenario where no GPUs are available or detected.
    Any method call attempting to interact with a GPU will raise a ValueError.
    """

    def __init__(self, ensure_homogeneous: bool = False) -> None:
        """Initialize the EMPTYGPUs class.

        Since this class represents an empty collection of GPUs, no actual initialization of GPU objects is performed.
        """
        pass

    def __del__(self) -> None:
        """Clean up any resources if necessary.

        As this class represents an empty collection of GPUs, no specific cleanup is required.
        """
        pass

    @property
    def gpus(self) -> Sequence["GPU"]:
        """Return an empty list as no GPUs are being tracked."""
        return []

    def __len__(self) -> int:
        """Return 0, indicating no GPUs are being tracked."""
        return 0

    def _ensure_homogeneous(self) -> None:
        """Raise a ValueError as no GPUs are being tracked."""
        raise ValueError("No GPUs available to ensure homogeneity.")

    def _warn_sys_admin(self) -> None:
        """Raise a ValueError as no GPUs are being tracked."""
        raise ValueError("No GPUs available to warn about SYS_ADMIN privileges.")

    @deprecated_alias("getName")
    def get_name(self, gpu_index: int) -> str:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    @deprecated_alias("getPowerManagementLimitConstraints")
    def get_power_management_limit_constraints(self, gpu_index: int) -> tuple[int, int]:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    def get_power_management_limit(self, gpu_index: int) -> int:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    @deprecated_alias("setPowerManagementLimit")
    def set_power_management_limit(self, gpu_index: int, power_limit_mw: int, block: bool = True) -> None:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    @deprecated_alias("resetPowerManagementLimit")
    def reset_power_management_limit(self, gpu_index: int, block: bool = True) -> None:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    @deprecated_alias("setPersistenceMode")
    def set_persistence_mode(self, gpu_index: int, enabled: bool, block: bool = True) -> None:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    @deprecated_alias("getSupportedMemoryClocks")
    def get_supported_memory_clocks(self, gpu_index: int) -> list[int]:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    @deprecated_alias("setMemoryLockedClocks")
    def set_memory_locked_clocks(
        self,
        gpu_index: int,
        min_clock_mhz: int,
        max_clock_mhz: int,
        block: bool = True,
    ) -> None:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    @deprecated_alias("resetMemoryLockedClocks")
    def reset_memory_locked_clocks(self, gpu_index: int, block: bool = True) -> None:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    @deprecated_alias("getSupportedGraphicsClocks")
    def get_supported_graphics_clocks(self, gpu_index: int, memory_clock_mhz: int | None = None) -> list[int]:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    @deprecated_alias("setGpuLockedClocks")
    def set_gpu_locked_clocks(
        self,
        gpu_index: int,
        min_clock_mhz: int,
        max_clock_mhz: int,
        block: bool = True,
    ) -> None:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    @deprecated_alias("resetGpuLockedClocks")
    def reset_gpu_locked_clocks(self, gpu_index: int, block: bool = True) -> None:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    @deprecated_alias("getInstantPowerUsage")
    def get_instant_power_usage(self, gpu_index: int) -> int:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    @deprecated_alias("supportsGetTotalEnergyConsumption")
    def supports_get_total_energy_consumption(self, gpu_index: int) -> bool:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    @deprecated_alias("getTotalEnergyConsumption")
    def get_total_energy_consumption(self, gpu_index: int) -> int:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    @deprecated_alias("getGpuTemperature")
    def get_gpu_temperature(self, gpu_index: int) -> int:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")


class ZeusGPUInitError(ZeusBaseGPUError):
    """Import error or GPU library initialization failures."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPUInvalidArgError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps Invalid Argument."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPUNotSupportedError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps Not Supported Operation on GPU."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPUNoPermissionError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps No Permission to perform GPU operation."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPUAlreadyInitializedError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps Already Initialized GPU."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPUNotFoundError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps Not Found GPU."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPUInsufficientSizeError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps Insufficient Size."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPUInsufficientPowerError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps Insufficient Power."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPUDriverNotLoadedError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps Driver Error."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPUTimeoutError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps Timeout Error."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPUIRQError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps IRQ Error."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPULibraryNotFoundError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps Library Not Found Error."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPUFunctionNotFoundError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps Function Not Found Error."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPUCorruptedInfoROMError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps Corrupted Info ROM Error."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPULostError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps Lost GPU Error."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPUResetRequiredError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps Reset Required Error."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPUOperatingSystemError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps Operating System Error."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPULibRMVersionMismatchError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps LibRM Version Mismatch Error."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPUMemoryError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps Insufficient Memory Error."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPUUnknownError(ZeusBaseGPUError):
    """Zeus GPU exception that wraps Unknown Error."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)


class ZeusGPUHeterogeneousError(ZeusBaseGPUError):
    """Exception for when GPUs are not homogeneous."""

    def __init__(self, message: str) -> None:
        """Intialize the exception object."""
        super().__init__(message)
