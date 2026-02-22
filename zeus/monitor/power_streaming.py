"""Stream GPU and CPU power readings from zeusd instances via SSE.

This module provides `PowerStreamingClient`, a thread-based SSE client
that connects to one or more zeusd endpoints (TCP or Unix domain socket)
and provides the latest GPU and CPU power readings in a thread-safe manner.

```python
from zeus.utils.zeusd import ZeusdConfig
from zeus.monitor.power_streaming import PowerStreamingClient

client = PowerStreamingClient(
    servers=[
        ZeusdConfig.tcp("node1", 4938, gpu_indices=[0, 1, 2, 3]),
        ZeusdConfig.tcp("node2", 4938),
    ],
)

# Snapshot (latest readings at this instant):
readings = client.get_power()  # {"node1:4938": PowerReadings(...)}

# Blocking iterator (yields on every SSE update):
for readings in client:
    print(readings)

# Async iterator:
async for readings in client:
    print(readings)

client.stop()
```
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
import time
import typing
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field

import httpx

from zeus.utils.zeusd import ZeusdCapabilityError, ZeusdClient, ZeusdConfig

logger = logging.getLogger(__name__)


@dataclass
class CpuPowerReading:
    """Power reading for a single CPU package.

    Args:
        cpu_w: CPU package power in watts.
        dram_w: DRAM power in watts, or None if not available.
    """

    cpu_w: float = 0.0
    dram_w: float | None = None


@dataclass
class PowerReadings:
    """Power readings from a single zeusd endpoint.

    Args:
        timestamp_s: Unix timestamp (seconds) of the reading.
        gpu_power_w: Per-GPU power in watts, keyed by GPU index.
        cpu_power_w: Per-CPU power readings, keyed by CPU index.
    """

    timestamp_s: float = 0.0
    gpu_power_w: dict[int, float] = field(default_factory=dict)
    cpu_power_w: dict[int, CpuPowerReading] = field(default_factory=dict)


class PowerStreamingClient:
    """Connect to multiple zeusd instances and stream GPU/CPU power readings.

    One background thread per device type per endpoint maintains an SSE
    connection to the zeusd streaming endpoints. The latest power readings
    are stored in a thread-safe dict, accessible via `get_power()`.

    The client supports three access patterns:

    - Snapshot: Call `get_power()` to retrieve the latest readings at any time.
    - Blocking iterator: Use `for readings in client` to block and yield a
      snapshot each time new SSE data arrives. Iteration stops when `stop()`
      is called.
    - Async iterator: Use `async for readings in client` for the same
      behavior without blocking the event loop.

    ```python
    client = PowerStreamingClient(servers=[...])

    # Snapshot
    readings = client.get_power()

    # Blocking iterator
    for readings in client:
        print(readings)

    # Async iterator
    async for readings in client:
        print(readings)

    client.stop()
    ```

    Args:
        servers: List of `ZeusdConfig` specifying zeusd endpoints
            and which GPUs/CPUs to stream from each.
        reconnect_delay_s: Seconds to wait before reconnecting after a
            disconnect.
    """

    def __init__(
        self,
        servers: Sequence[ZeusdConfig],
        reconnect_delay_s: float = 1.0,
    ) -> None:
        """Initialize the client and start background SSE connections."""
        self._servers = list(servers)
        self._reconnect_delay = reconnect_delay_s

        endpoints = [s.endpoint for s in self._servers]
        duplicates = [e for e in endpoints if endpoints.count(e) > 1]
        if duplicates:
            raise ValueError(f"Duplicate server endpoints: {sorted(set(duplicates))}")

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._readings: dict[str, PowerReadings] = {}
        self._clock_offsets: dict[str, float] = {}
        self._daemon_clients: dict[str, ZeusdClient] = {}

        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

        for server in self._servers:
            stream_gpu, stream_cpu = self._init_server(server)
            endpoint = server.endpoint
            self._clock_offsets[endpoint] = self._estimate_clock_offset(endpoint)

            if stream_gpu:
                t = threading.Thread(
                    target=self._gpu_stream_loop,
                    args=(server, endpoint),
                    name=f"gpu-power-stream-{endpoint}",
                    daemon=True,
                )
                t.start()
                self._threads.append(t)
                logger.info("Started GPU power streaming thread for %s", endpoint)

            if stream_cpu:
                t = threading.Thread(
                    target=self._cpu_stream_loop,
                    args=(server, endpoint),
                    name=f"cpu-power-stream-{endpoint}",
                    daemon=True,
                )
                t.start()
                self._threads.append(t)
                logger.info("Started CPU power streaming thread for %s", endpoint)

        if not self._threads:
            logger.warning(
                "No GPU or CPU power streaming threads were started. "
                "Check that the zeusd endpoints have the expected devices available."
            )

    def stop(self) -> None:
        """Stop all background connections and wake any blocked iterators."""
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads.clear()
        logger.info("All power streaming threads stopped")

    def get_power(self) -> dict[str, PowerReadings]:
        """Get the latest power readings from all endpoints.

        Returns:
            Mapping of endpoint identifier to `PowerReadings` containing
            timestamp and per-GPU/CPU power in watts.
        """
        with self._lock:
            return {
                k: PowerReadings(
                    timestamp_s=v.timestamp_s,
                    gpu_power_w=dict(v.gpu_power_w),
                    cpu_power_w={
                        idx: CpuPowerReading(cpu_w=r.cpu_w, dram_w=r.dram_w) for idx, r in v.cpu_power_w.items()
                    },
                )
                for k, v in self._readings.items()
            }

    def __iter__(self) -> Iterator[dict[str, PowerReadings]]:
        """Yield power reading snapshots as they arrive from SSE streams.

        Blocks until new readings are available, then yields a snapshot
        (same format as `get_power()`). Iteration stops when `stop()` is
        called.

        ```python
        client = PowerStreamingClient(servers=[...])
        for readings in client:
            for endpoint, pr in readings.items():
                print(f"{endpoint}: {pr.gpu_power_w}")
        ```
        """
        while not self._stop_event.is_set():
            with self._condition:
                notified = self._condition.wait(timeout=1.0)
            if self._stop_event.is_set():
                break
            if notified:
                yield self.get_power()

    async def __aiter__(self) -> AsyncIterator[dict[str, PowerReadings]]:
        """Async version of `__iter__`.

        Yields power reading snapshots without blocking the event loop.
        Iteration stops when `stop()` is called.

        ```python
        client = PowerStreamingClient(servers=[...])
        async for readings in client:
            for endpoint, pr in readings.items():
                print(f"{endpoint}: {pr.gpu_power_w}")
        ```
        """
        import asyncio

        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            notified = await loop.run_in_executor(None, self._wait_for_update)
            if self._stop_event.is_set():
                break
            if notified:
                yield self.get_power()

    def _wait_for_update(self) -> bool:
        """Block until readings are updated or timeout (1 s).

        Used by `__aiter__` to avoid blocking the async event loop.
        """
        with self._condition:
            return self._condition.wait(timeout=1.0)

    def _init_server(self, server: ZeusdConfig) -> tuple[bool, bool]:
        """Initialize connection to a server and decide what to stream.

        Creates a `ZeusdClient` for the server (handling discovery and auth),
        validates requested indices, and checks scope permissions.

        Returns:
            A `(stream_gpu, stream_cpu)` pair of booleans.

        Raises:
            ZeusdConnectionError: If the server is not reachable.
            ValueError: If explicitly requested indices are not available.
            ZeusdCapabilityError: If explicitly requested streaming requires
                a scope the token doesn't have.
        """
        client = ZeusdClient(server)
        endpoint = server.endpoint
        self._daemon_clients[endpoint] = client

        available_gpus = set(client.gpu_ids)
        available_cpus = set(client.cpu_ids)

        stream_gpu = self._resolve_streaming(
            user_indices=server.gpu_indices,
            available_ids=available_gpus,
            has_permission=client.can_read_gpu,
            scope_name="gpu-read",
            device_type="GPU",
            endpoint=endpoint,
        )

        stream_cpu = self._resolve_streaming(
            user_indices=server.cpu_indices,
            available_ids=available_cpus,
            has_permission=client.can_read_cpu,
            scope_name="cpu-read",
            device_type="CPU",
            endpoint=endpoint,
        )

        return stream_gpu, stream_cpu

    @staticmethod
    def _resolve_streaming(
        user_indices: list[int] | None,
        available_ids: set[int],
        has_permission: bool,
        scope_name: str,
        device_type: str,
        endpoint: str,
    ) -> bool:
        """Decide whether to stream a device type.

        Semantics:
        - `user_indices is None`: stream all available, silently skip if
          none exist or if the token lacks the scope.
        - `user_indices == []`: explicitly opt out, never stream.
        - `user_indices` is a non-empty list: require all IDs to exist
          and the scope to be granted; raise on mismatch.

        Returns:
            True if streaming should be started for this device type.

        Raises:
            ValueError: If explicit indices are not a subset of available.
            ZeusdCapabilityError: If explicit indices are given but the
                token lacks the required scope.
        """
        if user_indices is not None:
            if not user_indices:
                return False
            missing = set(user_indices) - available_ids
            if missing:
                raise ValueError(
                    f"{device_type} indices {sorted(missing)} requested for "
                    f"{endpoint} but only {sorted(available_ids)} are available"
                )
            if not has_permission:
                raise ZeusdCapabilityError(
                    f"Token for {endpoint} lacks required scope '{scope_name}' "
                    f"(explicitly requested {device_type.lower()}_indices={user_indices})"
                )
            return True

        if not available_ids:
            logger.info(
                "No %ss available on %s; skipping %s power streaming",
                device_type,
                endpoint,
                device_type,
            )
            return False

        if not has_permission:
            logger.info(
                "Token for %s lacks '%s' scope; skipping %s streaming",
                endpoint,
                scope_name,
                device_type,
            )
            return False

        return True

    def _estimate_clock_offset(
        self,
        endpoint: str,
        num_samples: int = 5,
    ) -> float:
        """Estimate the clock offset between this client and the daemon.

        Performs `num_samples` round-trips to `GET /time` on the daemon,
        computes `client_midpoint - daemon_time` for each, and returns the
        median offset in seconds. A positive offset means the daemon clock
        is behind the client clock.

        Args:
            endpoint: The endpoint identifier.
            num_samples: Number of round-trips for robustness.

        Returns:
            Estimated clock offset in seconds. Add this to daemon
            timestamps to align them with client time.
        """
        client = self._daemon_clients[endpoint]
        url = client.url("/time")
        offsets: list[float] = []
        with client.make_client() as http:
            for _ in range(num_samples):
                t1 = time.time()
                response = http.get(url)
                t2 = time.time()
                response.raise_for_status()
                daemon_time_s = response.json()["timestamp_ms"] / 1000.0
                client_midpoint_s = (t1 + t2) / 2.0
                offsets.append(client_midpoint_s - daemon_time_s)
        offset = statistics.median(offsets)
        logger.info(
            "Clock offset for %s: %.4f s (median of %d samples)",
            endpoint,
            offset,
            num_samples,
        )
        return offset

    def _gpu_stream_loop(self, server: ZeusdConfig, endpoint: str) -> None:
        """Background thread: stream GPU power from a single server."""
        client = self._daemon_clients[endpoint]
        base_url = client.url("/gpu/stream_power")
        if server.gpu_indices is not None:
            ids_param = ",".join(str(i) for i in server.gpu_indices)
            url = f"{base_url}?gpu_ids={ids_param}"
        else:
            url = base_url
        self._stream_loop(url, endpoint, self._process_gpu_event, "GPU")

    def _cpu_stream_loop(self, server: ZeusdConfig, endpoint: str) -> None:
        """Background thread: stream CPU power from a single server."""
        client = self._daemon_clients[endpoint]
        base_url = client.url("/cpu/stream_power")
        if server.cpu_indices is not None:
            ids_param = ",".join(str(i) for i in server.cpu_indices)
            url = f"{base_url}?cpu_ids={ids_param}"
        else:
            url = base_url
        self._stream_loop(url, endpoint, self._process_cpu_event, "CPU")

    def _stream_loop(
        self,
        url: str,
        endpoint: str,
        process_fn: typing.Callable[[str, str], None],
        label: str,
    ) -> None:
        """Shared reconnect loop for SSE streams."""
        while not self._stop_event.is_set():
            try:
                self._connect_and_stream(url, endpoint, process_fn)
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (401, 403):
                    logger.error(
                        "%s SSE connection to %s failed with HTTP %d: %s",
                        label,
                        endpoint,
                        e.response.status_code,
                        e.response.text,
                    )
                    return
                if not self._stop_event.is_set():
                    logger.warning(
                        "%s SSE connection to %s rejected (HTTP %d), reconnecting in %.1fs",
                        label,
                        endpoint,
                        e.response.status_code,
                        self._reconnect_delay,
                        exc_info=True,
                    )
                    self._stop_event.wait(timeout=self._reconnect_delay)
            except httpx.RequestError:
                if not self._stop_event.is_set():
                    logger.warning(
                        "%s SSE connection to %s lost, reconnecting in %.1fs",
                        label,
                        endpoint,
                        self._reconnect_delay,
                        exc_info=True,
                    )
                    self._stop_event.wait(timeout=self._reconnect_delay)

    def _connect_and_stream(
        self,
        url: str,
        endpoint: str,
        process_fn: typing.Callable[[str, str], None],
    ) -> None:
        """Open an SSE connection and process events until disconnected."""
        client = self._daemon_clients[endpoint]
        logger.info("Connecting to SSE at %s", url)
        with client.make_client() as http, http.stream("GET", url, timeout=None) as response:
            response.raise_for_status()
            logger.info("SSE connected to %s", url)
            buffer = ""
            for chunk in response.iter_text():
                if self._stop_event.is_set():
                    return
                buffer += chunk
                while "\n\n" in buffer:
                    event_text, buffer = buffer.split("\n\n", 1)
                    process_fn(event_text, endpoint)

    def _process_gpu_event(self, event_text: str, endpoint: str) -> None:
        """Parse a GPU SSE event and update readings."""
        for line in event_text.strip().split("\n"):
            if line.startswith("data: "):
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in GPU SSE event from %s: %s", endpoint, data_str[:100])
                    continue

                power_mw = data.get("power_mw", {})
                timestamp_ms = data.get("timestamp_ms", 0)
                timestamp_s = timestamp_ms / 1000.0 + self._clock_offsets[endpoint]

                gpu_power_w: dict[int, float] = {}
                for gpu_id_str, mw in power_mw.items():
                    gpu_power_w[int(gpu_id_str)] = float(mw) / 1000.0  # mW -> W

                with self._condition:
                    existing = self._readings.get(endpoint)
                    if existing is not None:
                        existing.gpu_power_w = gpu_power_w
                        existing.timestamp_s = max(existing.timestamp_s, timestamp_s)
                    else:
                        self._readings[endpoint] = PowerReadings(
                            timestamp_s=timestamp_s,
                            gpu_power_w=gpu_power_w,
                        )
                    self._condition.notify_all()

    def _process_cpu_event(self, event_text: str, endpoint: str) -> None:
        """Parse a CPU SSE event and update readings.

        Expected JSON format: `{"timestamp_ms": N, "power_mw": {"0": {"cpu_mw": N, "dram_mw": N|null}}}`.
        """
        for line in event_text.strip().split("\n"):
            if line.startswith("data: "):
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in CPU SSE event from %s: %s", endpoint, data_str[:100])
                    continue

                power_mw = data.get("power_mw", {})
                timestamp_ms = data.get("timestamp_ms", 0)
                timestamp_s = timestamp_ms / 1000.0 + self._clock_offsets[endpoint]

                cpu_power_w: dict[int, CpuPowerReading] = {}
                for cpu_id_str, readings in power_mw.items():
                    cpu_mw = readings.get("cpu_mw", 0)
                    dram_mw = readings.get("dram_mw")
                    cpu_power_w[int(cpu_id_str)] = CpuPowerReading(
                        cpu_w=float(cpu_mw) / 1000.0,
                        dram_w=float(dram_mw) / 1000.0 if dram_mw is not None else None,
                    )

                with self._condition:
                    existing = self._readings.get(endpoint)
                    if existing is not None:
                        existing.cpu_power_w = cpu_power_w
                        existing.timestamp_s = max(existing.timestamp_s, timestamp_s)
                    else:
                        self._readings[endpoint] = PowerReadings(
                            timestamp_s=timestamp_s,
                            cpu_power_w=cpu_power_w,
                        )
                    self._condition.notify_all()
