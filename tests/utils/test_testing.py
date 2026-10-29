"""Tests for the simulated devices in `zeus.utils.testing`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zeus.device import get_cpus, get_gpus
from zeus.device.cpu.common import CpuDramMeasurement
from zeus.device.gpu.common import ZeusGPUHeterogeneousError, ZeusGPUInvalidArgError, ZeusGPUNotSupportedError
from zeus.monitor import ZeusMonitor
from zeus.utils.testing import MockCPU, MockCPUs, MockGPU, MockGPUs

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class FakeClock:
    """A clock that only moves when the test sets `now`."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_gpu_energy_integrates_power_capped_at_the_power_limit() -> None:
    clock = FakeClock()
    gpu = MockGPU(power_mw=250_000, power_limit_constraints_mw=(100_000, 300_000), clock=clock)

    clock.now = 2.0
    assert gpu.get_total_energy_consumption() == 500_000  # 250 W for 2 s.

    # A limit below the power draw caps the power from now on.
    gpu.set_power_management_limit(200_000)
    assert gpu.get_instant_power_usage() == 200_000
    clock.now = 3.0
    assert gpu.get_total_energy_consumption() == 700_000  # Plus 200 W for 1 s.

    gpu.reset_power_management_limit()
    assert gpu.get_power_management_limit() == 300_000
    assert gpu.get_average_power_usage() == 250_000


def test_gpu_setters_change_state() -> None:
    gpu = MockGPU()

    gpu.set_persistence_mode(True)
    gpu.set_gpu_locked_clocks(900, 1500)
    gpu.set_memory_locked_clocks(1000, 1000)
    assert gpu.get_persistence_mode()
    assert gpu.gpu_locked_clocks_mhz == (900, 1500)
    assert gpu.memory_locked_clocks_mhz == (1000, 1000)

    gpu.reset_locked_clocks()
    assert gpu.gpu_locked_clocks_mhz is None
    assert gpu.memory_locked_clocks_mhz is None


def test_gpu_setters_reject_invalid_arguments() -> None:
    gpu = MockGPU(power_limit_constraints_mw=(100_000, 300_000))

    with pytest.raises(ZeusGPUInvalidArgError):
        gpu.set_power_management_limit(99_000)
    with pytest.raises(ZeusGPUInvalidArgError):
        gpu.set_power_management_limit(301_000)
    with pytest.raises(ZeusGPUInvalidArgError):
        gpu.set_gpu_locked_clocks(1500, 900)
    with pytest.raises(ZeusGPUInvalidArgError):
        gpu.set_memory_locked_clocks(1000, 500)

    assert gpu.get_power_management_limit() == 300_000
    assert gpu.gpu_locked_clocks_mhz is None
    assert gpu.memory_locked_clocks_mhz is None


def test_gpu_unsupported_queries_raise() -> None:
    gpu = MockGPU(supports_total_energy=False)

    assert not gpu.supports_get_total_energy_consumption()
    with pytest.raises(ZeusGPUNotSupportedError):
        gpu.get_total_energy_consumption()
    with pytest.raises(ZeusGPUNotSupportedError):
        gpu.get_average_memory_power_usage()

    assert MockGPU(memory_power_mw=20_000).get_average_memory_power_usage() == 20_000


def test_gpus_ensure_homogeneous() -> None:
    MockGPUs([MockGPU(0), MockGPU(1)], ensure_homogeneous=True)
    with pytest.raises(ZeusGPUHeterogeneousError):
        MockGPUs([MockGPU(0, name="A40"), MockGPU(1, name="V100")], ensure_homogeneous=True)


def test_cpu_energy_with_and_without_dram() -> None:
    clock = FakeClock()
    cpus = MockCPUs(
        [
            MockCPU(0, cpu_power_mw=50_000, dram_power_mw=10_000, clock=clock),
            MockCPU(1, cpu_power_mw=40_000, dram_power_mw=None, clock=clock),
        ]
    )

    clock.now = 2.0
    assert cpus.get_total_energy_consumption(0) == CpuDramMeasurement(cpu_mj=100_000.0, dram_mj=20_000.0)
    assert cpus.get_total_energy_consumption(1) == CpuDramMeasurement(cpu_mj=80_000.0, dram_mj=None)

    assert cpus.supports_get_dram_energy_consumption(0)
    assert not cpus.supports_get_dram_energy_consumption(1)
    assert cpus.get_power_limits(0).dram is not None
    assert cpus.get_power_limits(1).dram is None


def test_zeus_monitor_measures_simulated_devices(mocker: MockerFixture) -> None:
    """`ZeusMonitor` reads the simulated devices through `get_gpus` and `get_cpus`."""
    clock = FakeClock()
    gpus = MockGPUs([MockGPU(0, power_mw=200_000, clock=clock), MockGPU(1, power_mw=150_000, clock=clock)])
    cpus = MockCPUs([MockCPU(0, cpu_power_mw=50_000, dram_power_mw=10_000, clock=clock)])
    mocker.patch("zeus.device.gpu._gpus", gpus)
    mocker.patch("zeus.device.cpu._cpus", cpus)
    mocker.patch("zeus.monitor.energy.time", clock)

    assert get_gpus() is gpus
    assert get_cpus() is cpus

    monitor = ZeusMonitor()
    assert monitor.gpu_indices == [0, 1]
    assert monitor.cpu_indices == [0]

    monitor.begin_window("window", sync_execution=False)
    clock.now = 2.0
    measurement = monitor.end_window("window", sync_execution=False)

    assert measurement.time == 2.0
    assert measurement.gpu_energy == {0: 400.0, 1: 300.0}
    assert measurement.cpu_energy == {0: 100.0}
    assert measurement.dram_energy == {0: 20.0}
