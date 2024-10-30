"""Error wrappers and classes common to all GPU vendors."""

from __future__ import annotations

import abc
import warnings
from typing import Sequence

from zeus.device.exception import ZeusBaseGPUError
from zeus.utils.logging import get_logger
from zeus.device.common import has_sys_admin

logger = get_logger(__name__)


class GPU(abc.ABC):
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

    @abc.abstractmethod
    def getName(self) -> str:
        """Return the name of the GPU model."""
        pass

    @abc.abstractmethod
    def getPowerManagementLimitConstraints(self) -> tuple[int, int]:
        """Return the minimum and maximum power management limits. Units: mW."""
        pass

    @abc.abstractmethod
    def setPowerManagementLimit(self, power_limit_mw: int, _block: bool = True) -> None:
        """Set the GPU's power management limit. Unit: mW."""
        pass

    @abc.abstractmethod
    def resetPowerManagementLimit(self, _block: bool = True) -> None:
        """Reset the GPU's power management limit to the default value."""
        pass

    @abc.abstractmethod
    def setPersistenceMode(self, enabled: bool, _block: bool = True) -> None:
        """Set persistence mode."""
        pass

    @abc.abstractmethod
    def getSupportedMemoryClocks(self) -> list[int]:
        """Return a list of supported memory clock frequencies. Units: MHz."""
        pass

    @abc.abstractmethod
    def setMemoryLockedClocks(
        self, min_clock_mhz: int, max_clock_mhz: int, _block: bool = True
    ) -> None:
        """Lock the memory clock to a specified range. Units: MHz."""
        pass

    @abc.abstractmethod
    def resetMemoryLockedClocks(self, _block: bool = True) -> None:
        """Reset the locked memory clocks to the default."""
        pass

    @abc.abstractmethod
    def getSupportedGraphicsClocks(
        self, memory_clock_mhz: int | None = None
    ) -> list[int]:
        """Return a list of supported graphics clock frequencies. Units: MHz.

        Args:
            memory_clock_mhz: Memory clock frequency to use. Some GPUs have
                different supported graphics clocks depending on the memory clock.
        """
        pass

    @abc.abstractmethod
    def setGpuLockedClocks(
        self, min_clock_mhz: int, max_clock_mhz: int, _block: bool = True
    ) -> None:
        """Lock the GPU clock to a specified range. Units: MHz."""
        pass

    @abc.abstractmethod
    def resetGpuLockedClocks(self, _block: bool = True) -> None:
        """Reset the locked GPU clocks to the default."""
        pass

    @abc.abstractmethod
    def getAveragePowerUsage(self) -> int:
        """Return the average power usage of the GPU. Units: mW."""
        pass

    @abc.abstractmethod
    def getInstantPowerUsage(self) -> int:
        """Return the current power draw of the GPU. Units: mW."""
        pass

    @abc.abstractmethod
    def getAverageMemoryPowerUsage(self) -> int:
        """Return the average power usage of the GPU's memory. Units: mW."""
        pass

    @abc.abstractmethod
    def supportsGetTotalEnergyConsumption(self) -> bool:
        """Check if the GPU supports retrieving total energy consumption."""
        pass

    @abc.abstractmethod
    def getTotalEnergyConsumption(self) -> int:
        """Return the total energy consumption of the GPU since driver load. Units: mJ."""
        pass


class GPUs(abc.ABC):
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
        gpu_names = [gpu.getName() for gpu in self.gpus]
        # Both zero (no GPUs found) and one are fine.
        if len(set(gpu_names)) > 1:
            raise ZeusGPUHeterogeneousError(f"Heterogeneous GPUs found: {gpu_names}")

    def _warn_sys_admin(self) -> None:
        """Warn the user if the current process doesn't have `SYS_ADMIN` privileges."""
        # Deriving classes can disable this warning by setting this attribute.
        if (
            not getattr(self, "_disable_sys_admin_warning", False)
            and not has_sys_admin()
        ):
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

    def getName(self, gpu_index: int) -> str:
        """Return the name of the specified GPU."""
        return self.gpus[gpu_index].getName()

    def getPowerManagementLimitConstraints(self, gpu_index: int) -> tuple[int, int]:
        """Return the minimum and maximum power management limits. Units: mW."""
        return self.gpus[gpu_index].getPowerManagementLimitConstraints()

    def setPowerManagementLimit(
        self, gpu_index: int, power_limit_mw: int, _block: bool = True
    ) -> None:
        """Set the GPU's power management limit. Unit: mW."""
        self._warn_sys_admin()
        self.gpus[gpu_index].setPowerManagementLimit(power_limit_mw, _block)

    def resetPowerManagementLimit(self, gpu_index: int, _block: bool = True) -> None:
        """Reset the GPU's power management limit to the default value."""
        self._warn_sys_admin()
        self.gpus[gpu_index].resetPowerManagementLimit(_block)

    def setPersistenceMode(
        self, gpu_index: int, enabled: bool, _block: bool = True
    ) -> None:
        """Set persistence mode for the specified GPU."""
        self._warn_sys_admin()
        self.gpus[gpu_index].setPersistenceMode(enabled, _block)

    def getSupportedMemoryClocks(self, gpu_index: int) -> list[int]:
        """Return a list of supported memory clock frequencies. Units: MHz."""
        return self.gpus[gpu_index].getSupportedMemoryClocks()

    def setMemoryLockedClocks(
        self,
        gpu_index: int,
        min_clock_mhz: int,
        max_clock_mhz: int,
        _block: bool = True,
    ) -> None:
        """Lock the memory clock to a specified range. Units: MHz."""
        self._warn_sys_admin()
        self.gpus[gpu_index].setMemoryLockedClocks(min_clock_mhz, max_clock_mhz, _block)

    def resetMemoryLockedClocks(self, gpu_index: int, _block: bool = True) -> None:
        """Reset the locked memory clocks to the default."""
        self._warn_sys_admin()
        self.gpus[gpu_index].resetMemoryLockedClocks(_block)

    def getSupportedGraphicsClocks(
        self, gpu_index: int, memory_clock_mhz: int | None = None
    ) -> list[int]:
        """Return a list of supported graphics clock frequencies. Units: MHz.

        Args:
            gpu_index: Index of the GPU to query.
            memory_clock_mhz: Memory clock frequency to use. Some GPUs have
                different supported graphics clocks depending on the memory clock.
        """
        return self.gpus[gpu_index].getSupportedGraphicsClocks(memory_clock_mhz)

    def setGpuLockedClocks(
        self,
        gpu_index: int,
        min_clock_mhz: int,
        max_clock_mhz: int,
        _block: bool = True,
    ) -> None:
        """Lock the GPU clock to a specified range. Units: MHz."""
        self._warn_sys_admin()
        self.gpus[gpu_index].setGpuLockedClocks(min_clock_mhz, max_clock_mhz, _block)

    def resetGpuLockedClocks(self, gpu_index: int, _block: bool = True) -> None:
        """Reset the locked GPU clocks to the default."""
        self._warn_sys_admin()
        self.gpus[gpu_index].resetGpuLockedClocks(_block)

    def getInstantPowerUsage(self, gpu_index: int) -> int:
        """Return the current power draw of the GPU. Units: mW."""
        return self.gpus[gpu_index].getInstantPowerUsage()

    def getAverageMemoryPowerUsage(self, gpu_index: int) -> int:
        """Return the average power usage of the GPU's memory. Units: mW."""
        return self.gpus[gpu_index].getAverageMemoryPowerUsage()

    def supportsGetTotalEnergyConsumption(self, gpu_index: int) -> bool:
        """Check if the GPU supports retrieving total energy consumption."""
        return self.gpus[gpu_index].supportsGetTotalEnergyConsumption()

    def getTotalEnergyConsumption(self, gpu_index: int) -> int:
        """Return the total energy consumption of the GPU since driver load. Units: mJ."""
        return self.gpus[gpu_index].getTotalEnergyConsumption()


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

    def getName(self, gpu_index: int) -> str:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    def getPowerManagementLimitConstraints(self, gpu_index: int) -> tuple[int, int]:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    def setPowerManagementLimit(
        self, gpu_index: int, power_limit_mw: int, _block: bool = True
    ) -> None:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    def resetPowerManagementLimit(self, gpu_index: int, _block: bool = True) -> None:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    def setPersistenceMode(
        self, gpu_index: int, enabled: bool, _block: bool = True
    ) -> None:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    def getSupportedMemoryClocks(self, gpu_index: int) -> list[int]:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    def setMemoryLockedClocks(
        self,
        gpu_index: int,
        min_clock_mhz: int,
        max_clock_mhz: int,
        _block: bool = True,
    ) -> None:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    def resetMemoryLockedClocks(self, gpu_index: int, _block: bool = True) -> None:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    def getSupportedGraphicsClocks(
        self, gpu_index: int, memory_clock_mhz: int | None = None
    ) -> list[int]:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    def setGpuLockedClocks(
        self,
        gpu_index: int,
        min_clock_mhz: int,
        max_clock_mhz: int,
        _block: bool = True,
    ) -> None:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    def resetGpuLockedClocks(self, gpu_index: int, _block: bool = True) -> None:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    def getInstantPowerUsage(self, gpu_index: int) -> int:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    def supportsGetTotalEnergyConsumption(self, gpu_index: int) -> bool:
        """Raise a ValueError as no GPUs are available."""
        raise ValueError("No GPUs available.")

    def getTotalEnergyConsumption(self, gpu_index: int) -> int:
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
