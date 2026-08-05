"""Tests for the power monitor."""

from __future__ import annotations

import collections
import queue
from typing import TYPE_CHECKING

import pytest

from zeus.monitor.power import (
    PowerDomain,
    PowerMonitor,
    PowerSample,
    infer_counter_update_period,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def make_monitor(samples: dict[PowerDomain, dict[int, list[tuple[float, float]]]]) -> PowerMonitor:
    """Build a `PowerMonitor` with pre-enqueued samples, bypassing process spawning.

    Args:
        samples: Maps each monitored power domain to a dictionary mapping GPU
            indices to `(timestamp, power_mw)` sample tuples.
    """
    monitor = PowerMonitor.__new__(PowerMonitor)
    monitor.gpu_indices = sorted({gpu for per_gpu in samples.values() for gpu in per_gpu})
    monitor.measurement_domains = list(samples)
    monitor.data_queues = {domain: queue.Queue() for domain in samples}
    monitor.samples = {domain: {gpu: collections.deque() for gpu in per_gpu} for domain, per_gpu in samples.items()}
    for domain, per_gpu in samples.items():
        for gpu, entries in per_gpu.items():
            for ts, mw in entries:
                monitor.data_queues[domain].put(PowerSample(timestamp=ts, gpu_index=gpu, power_mw=mw))
    return monitor


def test_infers_half_the_fastest_counter_period(mocker: MockerFixture) -> None:
    """The fastest probed counter period is halved and returned.

    Regression test for the accumulator being seeded with `0.0` and folded with
    `min`, which pinned the result at `0.0` for every input.
    """
    gpus = mocker.MagicMock()
    gpus.get_name.side_effect = lambda index: ["A40", "V100"][index]
    mocker.patch("zeus.monitor.power.get_gpus", return_value=gpus)
    mocker.patch(
        "zeus.monitor.power._infer_counter_update_period_single",
        side_effect=lambda index: [0.4, 0.12][index],
    )

    # 0.12 s is the fastest counter, so poll twice per update at 0.06 s.
    assert infer_counter_update_period([0, 1]) == pytest.approx(0.06)


def test_get_power_defaults_to_device_instant() -> None:
    """Without an explicit domain, `get_power` reads device instant power."""
    monitor = make_monitor(
        {
            PowerDomain.DEVICE_INSTANT: {0: [(100.0, 50_000.0), (101.0, 60_000.0)]},
            PowerDomain.DEVICE_AVERAGE: {0: [(100.0, 70_000.0)]},
        }
    )
    assert monitor.get_power() == {0: 60.0}
    assert monitor.data_queues[PowerDomain.DEVICE_INSTANT].empty()
    assert not monitor.data_queues[PowerDomain.DEVICE_AVERAGE].empty()


def test_get_power_with_explicit_domain() -> None:
    """Both `PowerDomain` values and their string forms select the domain to read."""
    monitor = make_monitor({PowerDomain.DEVICE_AVERAGE: {0: [(100.0, 70_000.0), (101.0, 80_000.0)]}})
    assert monitor.get_power(power_domain=PowerDomain.DEVICE_AVERAGE) == {0: 80.0}
    assert monitor.get_power(power_domain="device_average") == {0: 80.0}
    assert monitor.get_power(time=100.2, power_domain="device_average") == {0: 70.0}


def test_get_power_unmonitored_domain_raises() -> None:
    """Querying a domain that is not monitored raises instead of falling back."""
    monitor = make_monitor({PowerDomain.DEVICE_AVERAGE: {0: [(100.0, 70_000.0)]}})
    with pytest.raises(ValueError, match="device_instant is not being monitored"):
        monitor.get_power()


def test_get_energy_auto_selects_domain() -> None:
    """`get_energy` prefers device instant power and falls back to device average."""
    monitor = make_monitor(
        {
            PowerDomain.DEVICE_INSTANT: {0: [(100.0, 100_000.0), (101.0, 100_000.0)]},
            PowerDomain.DEVICE_AVERAGE: {0: [(100.0, 200_000.0), (101.0, 200_000.0)]},
        }
    )
    assert monitor.get_energy(99.0, 102.0) == {0: pytest.approx(100.0)}

    average_only = make_monitor({PowerDomain.DEVICE_AVERAGE: {0: [(100.0, 200_000.0), (101.0, 200_000.0)]}})
    assert average_only.get_energy(99.0, 102.0) == {0: pytest.approx(200.0)}

    memory_only = make_monitor({PowerDomain.MEMORY_AVERAGE: {0: [(100.0, 10_000.0)]}})
    with pytest.raises(ValueError, match="Neither"):
        memory_only.get_energy(99.0, 102.0)


def test_get_energy_with_explicit_domain() -> None:
    """An explicit domain overrides auto-selection and unmonitored domains raise."""
    monitor = make_monitor(
        {
            PowerDomain.DEVICE_INSTANT: {0: [(100.0, 100_000.0), (101.0, 100_000.0)]},
            PowerDomain.DEVICE_AVERAGE: {0: [(100.0, 200_000.0), (101.0, 200_000.0)]},
        }
    )
    assert monitor.get_energy(99.0, 102.0, power_domain="device_average") == {0: pytest.approx(200.0)}
    with pytest.raises(ValueError, match="memory_average is not being monitored"):
        monitor.get_energy(99.0, 102.0, power_domain="memory_average")


def test_cli_power_queries_and_integrates_same_domain(mocker: MockerFixture) -> None:
    """The CLI power subcommand streams and integrates the domain the user selected."""
    from zeus.monitor.__main__ import power

    monitor = mocker.MagicMock()
    monitor.update_period = 0.1
    monitor.get_power.return_value = None
    monitor.get_energy.return_value = {0: 5.0}
    mocker.patch("zeus.monitor.__main__.PowerMonitor", return_value=monitor)
    mock_time = mocker.patch("zeus.monitor.__main__.time")
    mock_time.time.return_value = 0.0
    mock_time.sleep.side_effect = [None, KeyboardInterrupt]

    power(gpu_indices=[0], update_period=0.1, power_domain="device_average")

    monitor.get_power.assert_called_once_with(power_domain="device_average")
    monitor.get_energy.assert_called_once_with(0.0, 0.0, power_domain="device_average")
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
