"""Tests for PowerStreamingClient iteration interfaces."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from zeus.monitor.power_streaming import (
    CpuPowerReading,
    PowerReadings,
    PowerStreamingClient,
)


def _make_client() -> PowerStreamingClient:
    """Create a PowerStreamingClient without connecting to real servers."""
    client = PowerStreamingClient.__new__(PowerStreamingClient)
    client._servers = []
    client._reconnect_delay = 1.0
    client._lock = threading.Lock()
    client._condition = threading.Condition(client._lock)
    client._readings = {}
    client._clock_offsets = {}
    client._threads = []
    client._stop_event = threading.Event()
    return client


def _push_reading(
    client: PowerStreamingClient,
    key: str,
    gpu_power_w: dict[int, float],
    timestamp_s: float = 1.0,
) -> None:
    """Simulate a background thread pushing a GPU power reading."""
    with client._condition:
        existing = client._readings.get(key)
        if existing is not None:
            existing.gpu_power_w = gpu_power_w
            existing.timestamp_s = timestamp_s
        else:
            client._readings[key] = PowerReadings(
                timestamp_s=timestamp_s,
                gpu_power_w=dict(gpu_power_w),
            )
        client._condition.notify_all()


class TestIter:
    def test_yields_on_update(self) -> None:
        """__iter__ yields a snapshot when a reading arrives."""
        client = _make_client()

        def feeder() -> None:
            time.sleep(0.05)
            _push_reading(client, "node1:4938", {0: 350.0})
            time.sleep(0.05)
            client.stop()

        t = threading.Thread(target=feeder, daemon=True)
        t.start()

        results = list(client)
        t.join(timeout=5.0)

        assert len(results) == 1
        assert "node1:4938" in results[0]
        assert results[0]["node1:4938"].gpu_power_w[0] == 350.0

    def test_yields_multiple_updates(self) -> None:
        """__iter__ yields once per SSE event."""
        client = _make_client()

        def feeder() -> None:
            for i in range(3):
                time.sleep(0.05)
                _push_reading(client, "node1:4938", {0: float(100 * (i + 1))}, timestamp_s=float(i))
            time.sleep(0.05)
            client.stop()

        t = threading.Thread(target=feeder, daemon=True)
        t.start()

        results = list(client)
        t.join(timeout=5.0)

        assert len(results) == 3
        powers = [r["node1:4938"].gpu_power_w[0] for r in results]
        assert powers == [100.0, 200.0, 300.0]

    def test_stops_on_stop(self) -> None:
        """__iter__ exits promptly when stop() is called."""
        client = _make_client()

        def stopper() -> None:
            time.sleep(0.1)
            client.stop()

        t = threading.Thread(target=stopper, daemon=True)
        t.start()

        start = time.monotonic()
        results = list(client)
        elapsed = time.monotonic() - start

        t.join(timeout=5.0)

        assert results == []
        assert elapsed < 2.0

    def test_multiple_endpoints(self) -> None:
        """__iter__ snapshot includes readings from all endpoints."""
        client = _make_client()

        def feeder() -> None:
            time.sleep(0.05)
            _push_reading(client, "node1:4938", {0: 350.0})
            time.sleep(0.05)
            _push_reading(client, "node2:4938", {0: 400.0})
            time.sleep(0.05)
            client.stop()

        t = threading.Thread(target=feeder, daemon=True)
        t.start()

        results = list(client)
        t.join(timeout=5.0)

        assert len(results) == 2
        assert "node1:4938" in results[-1]
        assert "node2:4938" in results[-1]

    def test_snapshot_is_deep_copy(self) -> None:
        """Yielded snapshots are independent of internal state."""
        client = _make_client()

        def feeder() -> None:
            time.sleep(0.05)
            _push_reading(client, "node1:4938", {0: 100.0})
            time.sleep(0.05)
            _push_reading(client, "node1:4938", {0: 200.0})
            time.sleep(0.05)
            client.stop()

        t = threading.Thread(target=feeder, daemon=True)
        t.start()

        results = list(client)
        t.join(timeout=5.0)

        assert len(results) == 2
        assert results[0]["node1:4938"].gpu_power_w[0] == 100.0
        assert results[1]["node1:4938"].gpu_power_w[0] == 200.0


class TestAiter:
    def test_yields_on_update(self) -> None:
        """__aiter__ yields a snapshot when a reading arrives."""
        client = _make_client()

        def feeder() -> None:
            time.sleep(0.05)
            _push_reading(client, "node1:4938", {0: 350.0})
            time.sleep(0.05)
            client.stop()

        async def consume() -> list[dict[str, PowerReadings]]:
            t = threading.Thread(target=feeder, daemon=True)
            t.start()
            results = []
            async for readings in client:
                results.append(readings)
            t.join(timeout=5.0)
            return results

        results = asyncio.run(consume())
        assert len(results) == 1
        assert results[0]["node1:4938"].gpu_power_w[0] == 350.0

    def test_yields_multiple_updates(self) -> None:
        """__aiter__ yields once per SSE event."""
        client = _make_client()

        def feeder() -> None:
            for i in range(3):
                time.sleep(0.05)
                _push_reading(client, "node1:4938", {0: float(100 * (i + 1))}, timestamp_s=float(i))
            time.sleep(0.05)
            client.stop()

        async def consume() -> list[dict[str, PowerReadings]]:
            t = threading.Thread(target=feeder, daemon=True)
            t.start()
            results = []
            async for readings in client:
                results.append(readings)
            t.join(timeout=5.0)
            return results

        results = asyncio.run(consume())
        assert len(results) == 3
        powers = [r["node1:4938"].gpu_power_w[0] for r in results]
        assert powers == [100.0, 200.0, 300.0]

    def test_stops_on_stop(self) -> None:
        """__aiter__ exits promptly when stop() is called."""
        client = _make_client()

        def stopper() -> None:
            time.sleep(0.1)
            client.stop()

        async def consume() -> list[dict[str, PowerReadings]]:
            t = threading.Thread(target=stopper, daemon=True)
            t.start()
            results = []
            async for readings in client:
                results.append(readings)
            t.join(timeout=5.0)
            return results

        start = time.monotonic()
        results = asyncio.run(consume())
        elapsed = time.monotonic() - start

        assert results == []
        assert elapsed < 2.0

    def test_does_not_block_event_loop(self) -> None:
        """__aiter__ should not prevent other coroutines from running."""
        client = _make_client()
        other_ran = False

        async def other_task() -> None:
            nonlocal other_ran
            await asyncio.sleep(0.01)
            other_ran = True

        def feeder() -> None:
            time.sleep(0.1)
            client.stop()

        async def main() -> None:
            t = threading.Thread(target=feeder, daemon=True)
            t.start()
            task = asyncio.create_task(other_task())
            async for _ in client:
                pass
            await task
            t.join(timeout=5.0)

        asyncio.run(main())
        assert other_ran


class TestGetPower:
    def test_empty_before_any_reading(self) -> None:
        """get_power returns an empty dict before any readings arrive."""
        client = _make_client()
        assert client.get_power() == {}

    def test_returns_deep_copy(self) -> None:
        """Mutating the returned dict does not affect internal state."""
        client = _make_client()
        _push_reading(client, "node1:4938", {0: 350.0})

        readings1 = client.get_power()
        readings1["node1:4938"].gpu_power_w[0] = 999.0

        readings2 = client.get_power()
        assert readings2["node1:4938"].gpu_power_w[0] == 350.0

    def test_cpu_readings(self) -> None:
        """get_power correctly returns CPU readings."""
        client = _make_client()
        with client._condition:
            client._readings["node1:4938"] = PowerReadings(
                timestamp_s=1.0,
                cpu_power_w={0: CpuPowerReading(cpu_w=45.0, dram_w=12.5)},
            )
            client._condition.notify_all()

        readings = client.get_power()
        assert readings["node1:4938"].cpu_power_w[0].cpu_w == 45.0
        assert readings["node1:4938"].cpu_power_w[0].dram_w == 12.5


class TestClockOffset:
    def test_gpu_event_applies_offset(self) -> None:
        """_process_gpu_event adjusts timestamp by the clock offset."""
        client = _make_client()
        client._clock_offsets["node1:4938"] = 2.0

        client._process_gpu_event(
            'data: {"power_mw": {"0": 150000}, "timestamp_ms": 1000000}',
            "node1:4938",
        )

        readings = client._readings["node1:4938"]
        assert readings.timestamp_s == pytest.approx(1002.0)
        assert readings.gpu_power_w[0] == 150.0

    def test_cpu_event_applies_offset(self) -> None:
        """_process_cpu_event adjusts timestamp by the clock offset."""
        client = _make_client()
        client._clock_offsets["node1:4938"] = -1.5

        client._process_cpu_event(
            'data: {"power_mw": {"0": {"cpu_mw": 45000, "dram_mw": 12000}}, "timestamp_ms": 2000000}',
            "node1:4938",
        )

        readings = client._readings["node1:4938"]
        assert readings.timestamp_s == pytest.approx(1998.5)
        assert readings.cpu_power_w[0].cpu_w == 45.0
        assert readings.cpu_power_w[0].dram_w == 12.0

    def test_zero_offset_preserves_timestamp(self) -> None:
        """With zero offset, timestamps are unchanged."""
        client = _make_client()
        client._clock_offsets["node1:4938"] = 0.0

        client._process_gpu_event(
            'data: {"power_mw": {"0": 200000}, "timestamp_ms": 5000000}',
            "node1:4938",
        )

        assert client._readings["node1:4938"].timestamp_s == pytest.approx(5000.0)

    def test_max_timestamp_uses_adjusted_values(self) -> None:
        """When multiple events arrive, max() operates on offset-adjusted timestamps."""
        client = _make_client()
        client._clock_offsets["node1:4938"] = 10.0

        client._process_gpu_event(
            'data: {"power_mw": {"0": 100000}, "timestamp_ms": 1000}',
            "node1:4938",
        )
        client._process_gpu_event(
            'data: {"power_mw": {"0": 200000}, "timestamp_ms": 2000}',
            "node1:4938",
        )

        assert client._readings["node1:4938"].timestamp_s == pytest.approx(12.0)
