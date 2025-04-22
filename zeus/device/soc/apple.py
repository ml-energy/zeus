"""Apple Silicon SoC's."""

from __future__ import annotations

import sys
import platform
from dataclasses import dataclass, asdict, fields

from zeus.device.soc.common import SoC, SoCMeasurement, ZeusSoCInitError

# The following are optional dependencies. If a host machine does not have them
# installed, the Zeus code importing this module will gracefully handle the
# import error.
from zeus_apple_silicon import AppleEnergyMonitor, AppleEnergyMetrics  # type: ignore


class ZeusAppleInitError(ZeusSoCInitError):
    """Import error for Apple SoC initialization failures."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


@dataclass
class AppleSiliconMeasurement(SoCMeasurement):
    """Represents energy consumption of various subsystems on an Apple processor.

    All measurements are in mJ.
    """

    # CPU related metrics
    cpu_total_mj: int | None = None
    efficiency_cores_mj: list[int] | None = None
    performance_cores_mj: list[int] | None = None
    efficiency_core_manager_mj: int | None = None
    performance_core_manager_mj: int | None = None

    # DRAM
    dram_mj: int | None = None

    # GPU related metrics
    gpu_mj: int | None = None
    gpu_sram_mj: int | None = None

    # ANE (Apple Neural Engine)
    ane_mj: int | None = None

    def __sub__(self, other: "AppleSiliconMeasurement") -> "AppleSiliconMeasurement":
        """Produce a single measurement object containing differences across all fields."""
        if not isinstance(other, type(self)):
            raise TypeError(
                "Subtraction is only supported between AppleSiliconMeasurement instances."
            )

        result = self.__class__()

        for field in fields(self):
            f_name = field.name
            value1 = getattr(self, f_name, None)
            value2 = getattr(other, f_name, None)
            if value1 is None or value2 is None or type(value1) is not type(value2):
                continue

            if isinstance(value1, int):
                setattr(result, f_name, value1 - value2)
            elif isinstance(value1, list):
                if len(value1) != len(value2):
                    continue
                setattr(result, f_name, [x - y for x, y in zip(value1, value2)])

        return result

    def zeroAllFields(self) -> None:
        """Set the value of all fields in the measurement object to zero."""
        for field in fields(self):
            f_name = field.name
            setattr(self, f_name, 0)

        # Handle fields that are meant to be lists specially.
        list_fields = ["efficiency_cores_mj", "performance_cores_mj"]
        for f_name in list_fields:
            setattr(self, f_name, [])


def measurementFromMetrics(metrics: AppleEnergyMetrics) -> AppleSiliconMeasurement:
    """Return an AppleSiliconMeasurement object based on an AppleEnergyMetrics object."""
    return AppleSiliconMeasurement(
        cpu_total_mj=metrics.cpu_total_mj,
        efficiency_cores_mj=metrics.efficiency_cores_mj,
        performance_cores_mj=metrics.performance_cores_mj,
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
        self._monitor: AppleEnergyMonitor = None
        self.available_metrics: set[str] | None = None

        if sys.platform != "darwin" or platform.processor() != "arm":
            raise ZeusAppleInitError(
                "AppleSilicon is only supported on Apple silicon devices."
            )

        try:
            self._monitor = AppleEnergyMonitor()
        except RuntimeError as e:
            raise ZeusAppleInitError(
                f"Failed to initialize `AppleEnergyMonitor`: {e}"
            ) from None

    def getAvailableMetrics(self) -> set[str]:
        """Return a set of all observable metrics on the current processor."""
        if self.available_metrics is None:
            result: SoCMeasurement = self.getTotalEnergyConsumption()
            available_metrics = set()

            metrics_dict = asdict(result)
            for f_name, f_value in metrics_dict.items():
                if f_value is not None:
                    available_metrics.add(f"soc.{f_name}")

            self.available_metrics = available_metrics
        return self.available_metrics

    def getTotalEnergyConsumption(self) -> SoCMeasurement:
        """Returns the total energy consumption of the SoC.

        The measurement should be cumulative; different calls to this function throughout
        the lifetime of a single `SoC` manager object should count from a fixed arbitrary
        point in time.

        Units: mJ.
        """
        result: AppleEnergyMetrics = self._monitor.get_cumulative_energy()
        return measurementFromMetrics(result)

    def beginWindow(self, key) -> None:
        """Begin a measurement interval labeled with `key`."""
        self._monitor.begin_window(key)

    def endWindow(self, key) -> SoCMeasurement:
        """End a measurement window and return the energy consumption. Units: mJ."""
        result: AppleEnergyMetrics = self._monitor.end_window(key)
        return measurementFromMetrics(result)
