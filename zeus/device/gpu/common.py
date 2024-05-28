"""Error wrappers and classes common to all GPU vendors."""

from __future__ import annotations

import abc
from typing import Sequence

from zeus.device.exception import ZeusBaseGPUError


class ZeusGPUInitError(ZeusBaseGPUError):
    """Import error or GPU library initialization failures."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUInvalidArgError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Invalid Argument."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUNotSupportedError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Not Supported Operation on GPU."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUNoPermissionError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for No Permission to perform GPU operation."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUAlreadyInitializedError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Already Initialized GPU."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUNotFoundError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Not Found GPU."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUInsufficientSizeError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Insufficient Size."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUInsufficientPowerError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Insufficient Power."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUDriverNotLoadedError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Driver Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUTimeoutError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Timeout Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUIRQError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for IRQ Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPULibraryNotFoundError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Library Not Found Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUFunctionNotFoundError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Function Not Found Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUCorruptedInfoROMError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Corrupted Info ROM Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPULostError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Lost GPU Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUResetRequiredError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Reset Required Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUOperatingSystemError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Operating System Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPULibRMVersionMismatchError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for LibRM Version Mismatch Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUMemoryError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Insufficient Memory Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusGPUUnknownError(ZeusBaseGPUError):
    """Zeus GPU exception class Wrapper for Unknown Error."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class GPU(abc.ABC):
    """Abstract base class for GPU management.

    This class defines the interface for interacting with GPUs, subclasses should implement the methods to interact with specific GPU libraries
    (e.g., NVML for NVIDIA GPUs).
    """

    def __init__(self, gpu_index: int) -> None:
        """Initialize the GPU with a specified index."""
        self.gpu_index = gpu_index

    @abc.abstractmethod
    def getPowerManagementLimitConstraints(self) -> tuple[int, int]:
        """Return the minimum and maximum power management limits for the GPU. Units: mW."""
        pass

    @abc.abstractmethod
    def setPersistenceMode(self, enable: bool) -> None:
        """Enable persistence mode for the GPU."""
        pass

    @abc.abstractmethod
    def setPowerManagementLimit(self, value: int) -> None:
        """Set the power management limit for the GPU to a specified value or default. Unit: mW."""
        pass

    @abc.abstractmethod
    def resetPowerManagementLimit(self) -> None:
        """Resets the power management limit for the specified GPU to the default value."""
        pass

    @abc.abstractmethod
    def setMemoryLockedClocks(self, minMemClockMHz: int, maxMemClockMHz: int) -> None:
        """Lock the memory clock to a specified range. Units: MHz."""
        pass

    @abc.abstractmethod
    def getSupportedMemoryClocks(self) -> list[int]:
        """Return a list of supported memory clock frequencies for the GPU. Units: MHz."""
        pass

    @abc.abstractmethod
    def getSupportedGraphicsClocks(self, freq: int) -> list[int]:
        """Return a list of supported graphics clock frequencies for a given memory frequency. Units: MHz."""
        pass

    @abc.abstractmethod
    def getName(self) -> str:
        """Return the name of the GPU."""
        pass

    @abc.abstractmethod
    def setGpuLockedClocks(self, minGpuClockMHz: int, maxGpuClockMHz: int) -> None:
        """Lock the GPU clock to a specified range. Units: MHz."""
        pass

    @abc.abstractmethod
    def resetMemoryLockedClocks(self) -> None:
        """Reset the memory locked clocks to default values."""
        pass

    @abc.abstractmethod
    def resetGpuLockedClocks(self) -> None:
        """Reset the GPU locked clocks to default values."""
        pass

    @abc.abstractmethod
    def getInstantPowerUsage(self) -> int:
        """Returns the current power usage of the GPU. Units: mW."""
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
    """An abstract base class for GPU manager object.

    This class defines the essential interface and common functionality for GPU management, instantiating multiple `GPU` objects for each GPU being tracked.
    Forwards the call for a specific method to the corresponding GPU object.
    """

    @abc.abstractmethod
    def __init__(self, ensure_homogeneous: bool = False) -> None:
        """Initializes the GPU management library to communicate with the GPU driver and sets up tracking for specified GPUs."""
        pass

    @abc.abstractmethod
    def __del__(self) -> None:
        """Shuts down the GPU monitoring library to release resources and clean up."""
        pass

    @property
    @abc.abstractmethod
    def gpus(self) -> Sequence[GPU]:
        """Returns a list of GPU objects being tracked."""
        pass

    def _ensure_homogeneous(self) -> None:
        """Ensures that all tracked GPUs are homogeneous in terms of name."""
        gpu_names = [gpu.getName() for gpu in self.gpus]
        # Both zero (no GPUs found) and one are fine.
        if len(set(gpu_names)) > 1:
            raise ZeusBaseGPUError(f"Heterogeneous GPUs found: {gpu_names}")

    def getPowerManagementLimitConstraints(self, index: int) -> tuple[int, int]:
        """Returns the minimum and maximum power management limits for the specified GPU. Units: mW."""
        return self.gpus[index].getPowerManagementLimitConstraints()

    def setPersistenceMode(self, index: int, enable: bool) -> None:
        """Enables persistence mode for the specified GPU."""
        self.gpus[index].setPersistenceMode(enable)

    def setPowerManagementLimit(self, index: int, value: int) -> None:
        """Sets the power management limit for the specified GPU to the given value. Unit: mW."""
        self.gpus[index].setPowerManagementLimit(value)

    def resetPowerManagementLimit(self, index: int) -> None:
        """Resets the power management limit for the specified GPU to the default value."""
        self.gpus[index].resetPowerManagementLimit()

    def setMemoryLockedClocks(
        self, index: int, minMemClockMHz: int, maxMemClockMHz: int
    ) -> None:
        """Locks the memory clock of the specified GPU to a range defined by the minimum and maximum memory clock frequencies. Units: MHz."""
        self.gpus[index].setMemoryLockedClocks(minMemClockMHz, maxMemClockMHz)

    def getSupportedMemoryClocks(self, index: int) -> list[int]:
        """Returns a list of supported memory clock frequencies for the specified GPU. Units: MHz."""
        return self.gpus[index].getSupportedMemoryClocks()

    def getSupportedGraphicsClocks(self, index: int, freq: int) -> list[int]:
        """Returns a list of supported graphics clock frequencies for the specified GPU at a given frequency. Units: MHz."""
        return self.gpus[index].getSupportedGraphicsClocks(freq)

    def getName(self, index: int) -> str:
        """Returns the name of the specified GPU."""
        return self.gpus[index].getName()

    def setGpuLockedClocks(
        self, index: int, minGpuClockMHz: int, maxGpuClockMHz: int
    ) -> None:
        """Locks the GPU clock of the specified GPU to a range defined by the minimum and maximum GPU clock frequencies. Units: MHz."""
        self.gpus[index].setGpuLockedClocks(minGpuClockMHz, maxGpuClockMHz)

    def resetMemoryLockedClocks(self, index: int) -> None:
        """Resets the memory locked clocks of the specified GPU to their default values."""
        self.gpus[index].resetMemoryLockedClocks()

    def resetGpuLockedClocks(self, index: int) -> None:
        """Resets the GPU locked clocks of the specified GPU to their default values."""
        self.gpus[index].resetGpuLockedClocks()

    def getInstantPowerUsage(self, index: int) -> int:
        """Returns the power usage of the specified GPU. Units: mW."""
        return self.gpus[index].getInstantPowerUsage()

    def supportsGetTotalEnergyConsumption(self, index: int) -> bool:
        """Returns True if the specified GPU supports retrieving the total energy consumption."""
        return self.gpus[index].supportsGetTotalEnergyConsumption()

    def getTotalEnergyConsumption(self, index: int) -> int:
        """Returns the total energy consumption of the specified GPU. Units: mJ."""
        return self.gpus[index].getTotalEnergyConsumption()

    def __len__(self) -> int:
        """Returns the number of GPUs being tracked."""
        return len(self.gpus)
