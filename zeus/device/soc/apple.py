"""Apple Silicon SoC's."""

from __future__ import annotations

import sys
import platform
import importlib
from dataclasses import dataclass, asdict, fields
from functools import lru_cache
from typing import Any

from zeus.device.soc.common import SoC, SoCMeasurement, ZeusSoCInitError

# `zeus_apple_silicon` is a Mac-only extra and is never installable on
# Linux. Loading it via `importlib.import_module` (rather than `import`)
# keeps `ty`'s static view identical across platforms — there is no
# `import` statement for it to resolve, so we don't need conflicting
# ignore comments for the Linux (unresolved import) and macOS (real
# module type vs. mock fallback) cases.
zeus_apple_silicon: Any
zeus_apple_available: bool

try:
    zeus_apple_silicon = importlib.import_module("zeus_apple_silicon")
    zeus_apple_available = True

except Exception:

    class MockZeusAppleSilicon:
        """Mock class for zeus-apple-silicon library."""

        def __getattr__(self, name):
            """Raise an error if any method is called.

            Since this class is only used when `zeus-apple-silicon` is not
            available, something has gone wrong if any method is called.
            """
            raise RuntimeError(
                f"zeus-apple-silicon is not available and zeus-apple-silicon.{name} "
                "shouldn't have been called. This is a bug."
            )

    zeus_apple_silicon = MockZeusAppleSilicon()
    zeus_apple_available = False


@lru_cache(maxsize=1)
def apple_silicon_is_available() -> bool:
    """Check if Apple silicon is available."""
    if not zeus_apple_available:
        return False
    if sys.platform != "darwin" or platform.processor() != "arm":
        return False
    return True


class ZeusAppleInitError(ZeusSoCInitError):
    """Import error for Apple SoC initialization failures."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


@dataclass
class AppleSiliconMeasurement(SoCMeasurement):
    """Represents energy consumption of various subsystems on an Apple processor.

    All measurements are in mJ. Fields that are unavailable on the current
    processor will be `None`.

    Attributes:
        cpu_total_mj: Total energy consumed by all CPU subsystems combined.
        efficiency_cores_mj: Per-core energy for each efficiency core.
        performance_cores_mj: Per-core energy for each performance core.
        efficiency_cluster_mj: Per-cluster energy for efficiency core clusters,
            including shared resources like L2 cache. More representative of
            end-to-end energy than the sum of individual cores.
        performance_cluster_mj: Per-cluster energy for performance core clusters,
            including shared resources like L2 cache. More representative of
            end-to-end energy than the sum of individual cores.
        efficiency_core_manager_mj: Energy for efficiency core cluster management.
        performance_core_manager_mj: Energy for performance core cluster management.
        dram_mj: Energy consumed by DRAM.
        gpu_mj: Energy consumed by the on-chip GPU.
        gpu_sram_mj: Energy consumed by GPU SRAM.
        ane_mj: Energy consumed by the Apple Neural Engine.
    """

    cpu_total_mj: int | None = None
    efficiency_cores_mj: list[int] | None = None
    performance_cores_mj: list[int] | None = None
    efficiency_cluster_mj: list[int] | None = None
    performance_cluster_mj: list[int] | None = None
    efficiency_core_manager_mj: int | None = None
    performance_core_manager_mj: int | None = None
    dram_mj: int | None = None
    gpu_mj: int | None = None
    gpu_sram_mj: int | None = None
    ane_mj: int | None = None

    def __sub__(self, other: AppleSiliconMeasurement) -> AppleSiliconMeasurement:
        """Produce a single measurement object containing differences across all fields."""
        if not isinstance(other, type(self)):
            raise TypeError("Subtraction is only supported between AppleSiliconMeasurement instances.")

        result = self.__class__()

        for field in fields(self):
            f_name = field.name
            value1 = getattr(self, f_name)
            value2 = getattr(other, f_name)
            if value1 is None and value2 is None:
                continue

            if type(value1) is not type(value2):
                raise ValueError(f"Inconsistent field between two AppleSiliconMeasurement objects: {f_name}")

            if isinstance(value1, int):
                setattr(result, f_name, value1 - value2)
            elif isinstance(value1, list):
                if len(value1) != len(value2):
                    raise ValueError(f"Inconsistent field between two AppleSiliconMeasurement objects: {f_name}")
                setattr(result, f_name, [x - y for x, y in zip(value1, value2)])

        return result

    def zero_all_fields(self) -> None:
        """Set the value of all fields in the measurement object to zero."""
        for field in fields(self):
            f_name = field.name
            f_value = getattr(self, f_name)
            if isinstance(f_value, int):
                setattr(self, f_name, 0)
            elif isinstance(f_value, list):
                setattr(self, f_name, [])
            else:
                setattr(self, f_name, None)

    @classmethod
    def from_metrics(
        cls,
        metrics: zeus_apple_silicon.AppleEnergyMetrics,
    ) -> AppleSiliconMeasurement:
        """Return an AppleSiliconMeasurement object based on an AppleEnergyMetrics object."""
        return cls(
            cpu_total_mj=metrics.cpu_total_mj,
            efficiency_cores_mj=metrics.efficiency_cores_mj,
            performance_cores_mj=metrics.performance_cores_mj,
            efficiency_cluster_mj=metrics.efficiency_cluster_mj,
            performance_cluster_mj=metrics.performance_cluster_mj,
            efficiency_core_manager_mj=metrics.efficiency_core_manager_mj,
            performance_core_manager_mj=metrics.performance_core_manager_mj,
            dram_mj=metrics.dram_mj,
            gpu_mj=metrics.gpu_mj,
            gpu_sram_mj=metrics.gpu_sram_mj,
            ane_mj=metrics.ane_mj,
        )


class AppleSilicon(SoC):
    """An interface for obtaining energy metrics of an Apple processor."""

    def __init__(self) -> None:
        """Initialize an instance of an Apple Silicon energy monitor."""
        self._monitor: zeus_apple_silicon.AppleEnergyMonitor
        self.available_metrics: set[str] | None = None

        try:
            self._monitor = zeus_apple_silicon.AppleEnergyMonitor()

        # This except block exists for failures the AppleEnergyMonitor
        # object may encounter during its own construction.
        except RuntimeError as e:
            raise ZeusAppleInitError(f"Failed to initialize `AppleEnergyMonitor`: {e}") from None

    def get_available_metrics(self) -> set[str]:
        """Return a set of all observable metrics on the current processor."""
        if self.available_metrics is None:
            result: SoCMeasurement = self.get_total_energy_consumption()
            available_metrics = set()

            metrics_dict = asdict(result)
            for f_name, f_value in metrics_dict.items():
                if f_value is not None:
                    available_metrics.add(f_name)

            self.available_metrics = available_metrics
        return self.available_metrics

    def get_total_energy_consumption(self) -> AppleSiliconMeasurement:
        """Returns the total energy consumption of the SoC.

        The measurement should be cumulative; different calls to this function throughout
        the lifetime of a single `SoC` manager object should count from a fixed arbitrary
        point in time.

        Units: mJ.
        """
        result = self._monitor.get_cumulative_energy()
        return AppleSiliconMeasurement.from_metrics(result)

    def begin_window(self, key: str, restart: bool = False) -> None:
        """Begin a measurement interval labeled with `key`.

        Args:
            key: Unique name of the measurement window.
            restart: If True and the window already exists, cancel the existing
                window and start a new one.
        """
        self._monitor.begin_window(key, restart=restart)

    def end_window(self, key: str) -> AppleSiliconMeasurement:
        """End a measurement window and return the energy consumption. Units: mJ."""
        result = self._monitor.end_window(key)
        return AppleSiliconMeasurement.from_metrics(result)
