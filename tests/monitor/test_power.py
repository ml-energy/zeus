from __future__ import annotations

from queue import Empty

from zeus.device.gpu.common import ZeusGPUInitError, ZeusGPUNotSupportedError
from zeus.monitor.power import PowerDomain, PowerMonitor


class FakeQueue:
    def get_nowait(self):
        raise Empty


class FakeEvent:
    def __init__(self) -> None:
        self._is_set = False

    def set(self) -> None:
        self._is_set = True

    def wait(self, timeout: float | None = None) -> bool:
        return self._is_set

    def is_set(self) -> bool:
        return self._is_set


class FakeProcess:
    def __init__(self, *, target, kwargs, daemon, name) -> None:
        self._kwargs = kwargs
        self.daemon = daemon
        self.name = name
        self._alive = False

    def start(self) -> None:
        self._alive = True
        self._kwargs["ready_event"].set()

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self._alive = False

    def terminate(self) -> None:
        self._alive = False

    def kill(self) -> None:
        self._alive = False


class FakeContext:
    def Queue(self):
        return FakeQueue()

    def Event(self):
        return FakeEvent()

    def Process(self, **kwargs):
        return FakeProcess(**kwargs)


class MockCPUs:
    def __init__(self, count: int) -> None:
        self._count = count

    def __len__(self) -> int:
        return self._count

    def supports_get_dram_energy_consumption(self, index: int) -> bool:
        return index == 0


class MockGPUs:
    def __init__(self, count: int) -> None:
        self._count = count

    def __len__(self) -> int:
        return self._count

    def get_instant_power_usage(self, index: int) -> int:
        raise ZeusGPUNotSupportedError("Unsupported")

    def get_average_power_usage(self, index: int) -> int:
        raise ZeusGPUNotSupportedError("Unsupported")

    def get_average_memory_power_usage(self, index: int) -> int:
        raise ZeusGPUNotSupportedError("Unsupported")


def test_none_selects_all_available_devices(mocker) -> None:
    mocker.patch("zeus.monitor.power.get_gpus", return_value=MockGPUs(count=2))
    mocker.patch("zeus.monitor.power.get_cpus", return_value=MockCPUs(count=2))
    mocker.patch("zeus.monitor.power.mp.get_context", return_value=FakeContext())

    monitor = PowerMonitor(update_period=0.1)

    try:
        assert monitor.gpu_indices == [0, 1]
        assert monitor.cpu_indices == [0, 1]
        assert set(monitor.cpu_measurement_domains) == {
            PowerDomain.CPU_PACKAGE_AVERAGE,
            PowerDomain.CPU_DRAM_AVERAGE,
        }
    finally:
        monitor.stop()


def test_cpu_only_monitor_handles_an_unavailable_gpu_backend(mocker) -> None:
    mocker.patch(
        "zeus.monitor.power.get_gpus",
        side_effect=ZeusGPUInitError("No GPU backend is available."),
    )
    mocker.patch("zeus.monitor.power.get_cpus", return_value=MockCPUs(count=2))
    mocker.patch("zeus.monitor.power.mp.get_context", return_value=FakeContext())

    monitor = PowerMonitor(update_period=0.1)

    try:
        assert monitor.gpu_indices == []
        assert monitor.cpu_indices == [0, 1]
        assert monitor.measurement_domains == []
        assert set(monitor.cpu_measurement_domains) == {
            PowerDomain.CPU_PACKAGE_AVERAGE,
            PowerDomain.CPU_DRAM_AVERAGE,
        }
    finally:
        monitor.stop()


def test_empty_cpu_indices_disable_cpu_measurement(mocker) -> None:
    mocker.patch("zeus.monitor.power.get_gpus", return_value=MockGPUs(count=1))
    get_cpus = mocker.patch(
        "zeus.monitor.power.get_cpus",
        return_value=MockCPUs(count=2),
    )

    monitor = PowerMonitor(cpu_indices=[], update_period=0.1)

    try:
        assert monitor.cpu_indices == []
        assert monitor.cpu_measurement_domains == []
        get_cpus.assert_called_once_with()
    finally:
        monitor.stop()
