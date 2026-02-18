"""Stream GPU and CPU power readings from zeusd instances via SSE.

This module provides `PowerStreamingClient`, a thread-based SSE client
that connects to one or more zeusd endpoints (TCP or Unix domain socket)
and provides the latest GPU and CPU power readings in a thread-safe manner.

```python
from zeus.monitor.power_streaming import PowerStreamingClient, ZeusdTcpConfig

client = PowerStreamingClient(
    servers=[
        ZeusdTcpConfig(
            host="node1", port=4938,
            gpu_indices=[0, 1, 2, 3],
        ),
        ZeusdTcpConfig(host="node2", port=4938),
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
import threading
import typing
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZeusdTcpConfig:
    """Connection configuration for a zeusd instance over TCP.

    Args:
        host: Hostname or IP of the zeusd instance.
        port: TCP port of the zeusd instance.
        gpu_indices: GPU device indices to stream. If None, all GPUs on
            the endpoint are streamed. Pass an empty list to skip GPU streaming.
        cpu_indices: CPU device indices to stream. If None, all CPUs on
            the endpoint are streamed (when RAPL is available). Pass an
            empty list to skip CPU streaming.
    """

    host: str
    port: int = 4938
    gpu_indices: list[int] | None = None
    cpu_indices: list[int] | None = None

    @property
    def key(self) -> str:
        """Return the `host:port` identifier for this server."""
        return f"{self.host}:{self.port}"


@dataclass(frozen=True)
class ZeusdUdsConfig:
    """Connection configuration for a zeusd instance over a Unix domain socket.

    Args:
        socket_path: Path to the zeusd Unix domain socket.
        gpu_indices: GPU device indices to stream. If None, all GPUs on
            the endpoint are streamed. Pass an empty list to skip GPU streaming.
        cpu_indices: CPU device indices to stream. If None, all CPUs on
            the endpoint are streamed (when RAPL is available). Pass an
            empty list to skip CPU streaming.
    """

    socket_path: str
    gpu_indices: list[int] | None = None
    cpu_indices: list[int] | None = None

    @property
    def key(self) -> str:
        """Return the socket path identifier for this server."""
        return self.socket_path


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
        servers: List of `ZeusdTcpConfig` or `ZeusdUdsConfig` specifying
            zeusd endpoints and which GPUs/CPUs to stream from each.
        reconnect_delay_s: Seconds to wait before reconnecting after a
            disconnect.
    """

    def __init__(
        self,
        servers: Sequence[ZeusdTcpConfig | ZeusdUdsConfig],
        reconnect_delay_s: float = 1.0,
    ) -> None:
        """Initialize the client and start background SSE connections."""
        self._servers = list(servers)
        self._reconnect_delay = reconnect_delay_s

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._readings: dict[str, PowerReadings] = {}

        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

        for server in self._servers:
            self._check_server_reachable(server)

            if server.gpu_indices is None or server.gpu_indices:
                t = threading.Thread(
                    target=self._gpu_stream_loop,
                    args=(server,),
                    name=f"gpu-power-stream-{server.key}",
                    daemon=True,
                )
                t.start()
                self._threads.append(t)
                logger.info("Started GPU power streaming thread for %s", server.key)

            if server.cpu_indices is None or server.cpu_indices:
                if self._check_cpu_available(server):
                    t = threading.Thread(
                        target=self._cpu_stream_loop,
                        args=(server,),
                        name=f"cpu-power-stream-{server.key}",
                        daemon=True,
                    )
                    t.start()
                    self._threads.append(t)
                    logger.info("Started CPU power streaming thread for %s", server.key)
                else:
                    raise ValueError(
                        f"CPU power streaming requested for {server.key} but RAPL is not "
                        "available on that server; skipping CPU streaming",
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
            Mapping of `"host:port"` to `PowerReadings` containing
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
            for key, pr in readings.items():
                print(f"{key}: {pr.gpu_power_w}")
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
            for key, pr in readings.items():
                print(f"{key}: {pr.gpu_power_w}")
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

    def _make_http_client(
        self,
        server: ZeusdTcpConfig | ZeusdUdsConfig,
        **kwargs: typing.Any,
    ) -> httpx.Client:
        """Create an httpx Client configured for the server's transport."""
        if isinstance(server, ZeusdUdsConfig):
            transport = httpx.HTTPTransport(uds=server.socket_path)
            return httpx.Client(transport=transport, **kwargs)
        return httpx.Client(**kwargs)

    def _url(self, server: ZeusdTcpConfig | ZeusdUdsConfig, path: str) -> str:
        """Build the full URL for the given server and path."""
        if isinstance(server, ZeusdUdsConfig):
            return f"http://localhost{path}"
        return f"http://{server.host}:{server.port}{path}"

    def _check_server_reachable(self, server: ZeusdTcpConfig | ZeusdUdsConfig) -> None:
        """Probe the server and validate requested device indices.

        Raises:
            ConnectionError: If the server is not reachable.
            ValueError: If requested GPU or CPU indices are not available.
        """
        url = self._url(server, "/gpu/power")
        try:
            with self._make_http_client(server, timeout=5.0) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()
        except httpx.RequestError as e:
            raise ConnectionError(
                f"Cannot reach zeusd at {server.key}. Is zeusd running and accessible at this address?"
            ) from e
        except httpx.HTTPStatusError as e:
            raise ConnectionError(f"zeusd at {server.key} returned HTTP {e.response.status_code}") from e

        if server.gpu_indices is not None:
            available = {int(k) for k in data.get("power_mw", {})}
            requested = set(server.gpu_indices)
            missing = requested - available
            if missing:
                raise ValueError(
                    f"GPU indices {sorted(missing)} requested for {server.key} "
                    f"but only {sorted(available)} are available"
                )

    def _check_cpu_available(self, server: ZeusdTcpConfig | ZeusdUdsConfig) -> bool:
        """Probe the one-shot CPU power endpoint to check RAPL availability.

        Raises:
            ValueError: If requested CPU indices are not available on the server.
        """
        url = self._url(server, "/cpu/power")
        try:
            with self._make_http_client(server, timeout=5.0) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()
                power_mw = data.get("power_mw", {})
                if power_mw:
                    if server.cpu_indices is not None:
                        available = {int(k) for k in power_mw}
                        requested = set(server.cpu_indices)
                        missing = requested - available
                        if missing:
                            raise ValueError(
                                f"CPU indices {sorted(missing)} requested for "
                                f"{server.key} but only {sorted(available)} "
                                f"are available"
                            )
                    return True
                return False
        except (httpx.RequestError, httpx.HTTPStatusError):
            logger.warning("Failed to probe CPU power endpoint on %s", server.key, exc_info=True)
            return False

    def _gpu_stream_loop(self, server: ZeusdTcpConfig | ZeusdUdsConfig) -> None:
        """Background thread: stream GPU power from a single server."""
        base_url = self._url(server, "/gpu/power/stream")
        # User specified specific indices to stream
        if server.gpu_indices is not None:
            ids_param = ",".join(str(i) for i in server.gpu_indices)
            url = f"{base_url}?gpu_ids={ids_param}"
        # User wants all available indices
        else:
            url = base_url
        self._stream_loop(url, server, self._process_gpu_event, "GPU")

    def _cpu_stream_loop(self, server: ZeusdTcpConfig | ZeusdUdsConfig) -> None:
        """Background thread: stream CPU power from a single server."""
        base_url = self._url(server, "/cpu/power/stream")
        # User specified specific indices to stream
        if server.cpu_indices is not None:
            ids_param = ",".join(str(i) for i in server.cpu_indices)
            url = f"{base_url}?cpu_ids={ids_param}"
        # User wants all available indices
        else:
            url = base_url
        self._stream_loop(url, server, self._process_cpu_event, "CPU")

    def _stream_loop(
        self,
        url: str,
        server: ZeusdTcpConfig | ZeusdUdsConfig,
        process_fn: typing.Callable[[str, str], None],
        label: str,
    ) -> None:
        """Shared reconnect loop for SSE streams."""
        while not self._stop_event.is_set():
            try:
                self._connect_and_stream(url, server, process_fn)
            except httpx.HTTPStatusError:
                if not self._stop_event.is_set():
                    logger.warning(
                        "%s SSE connection to %s rejected, reconnecting in %.1fs",
                        label,
                        server.key,
                        self._reconnect_delay,
                        exc_info=True,
                    )
                    self._stop_event.wait(timeout=self._reconnect_delay)
            except httpx.RequestError:
                if not self._stop_event.is_set():
                    logger.warning(
                        "%s SSE connection to %s lost, reconnecting in %.1fs",
                        label,
                        server.key,
                        self._reconnect_delay,
                        exc_info=True,
                    )
                    self._stop_event.wait(timeout=self._reconnect_delay)

    def _connect_and_stream(
        self,
        url: str,
        server: ZeusdTcpConfig | ZeusdUdsConfig,
        process_fn: typing.Callable[[str, str], None],
    ) -> None:
        """Open an SSE connection and process events until disconnected."""
        logger.info("Connecting to SSE at %s", url)
        with self._make_http_client(server, timeout=None) as client, client.stream("GET", url) as response:
            response.raise_for_status()
            logger.info("SSE connected to %s", url)
            buffer = ""
            for chunk in response.iter_text():
                if self._stop_event.is_set():
                    return
                buffer += chunk
                while "\n\n" in buffer:
                    event_text, buffer = buffer.split("\n\n", 1)
                    process_fn(event_text, server.key)

    def _process_gpu_event(self, event_text: str, key: str) -> None:
        """Parse a GPU SSE event and update readings."""
        for line in event_text.strip().split("\n"):
            if line.startswith("data: "):
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in GPU SSE event from %s: %s", key, data_str[:100])
                    continue

                power_mw = data.get("power_mw", {})
                timestamp_ms = data.get("timestamp_ms", 0)

                gpu_power_w: dict[int, float] = {}
                for gpu_id_str, mw in power_mw.items():
                    gpu_power_w[int(gpu_id_str)] = float(mw) / 1000.0  # mW -> W

                with self._condition:
                    existing = self._readings.get(key)
                    if existing is not None:
                        existing.gpu_power_w = gpu_power_w
                        existing.timestamp_s = max(existing.timestamp_s, timestamp_ms / 1000.0)
                    else:
                        self._readings[key] = PowerReadings(
                            timestamp_s=timestamp_ms / 1000.0,
                            gpu_power_w=gpu_power_w,
                        )
                    self._condition.notify_all()

    def _process_cpu_event(self, event_text: str, key: str) -> None:
        """Parse a CPU SSE event and update readings.

        Expected JSON format: `{"timestamp_ms": N, "power_mw": {"0": {"cpu_mw": N, "dram_mw": N|null}}}`.
        """
        for line in event_text.strip().split("\n"):
            if line.startswith("data: "):
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in CPU SSE event from %s: %s", key, data_str[:100])
                    continue

                power_mw = data.get("power_mw", {})
                timestamp_ms = data.get("timestamp_ms", 0)

                cpu_power_w: dict[int, CpuPowerReading] = {}
                for cpu_id_str, readings in power_mw.items():
                    cpu_mw = readings.get("cpu_mw", 0)
                    dram_mw = readings.get("dram_mw")
                    cpu_power_w[int(cpu_id_str)] = CpuPowerReading(
                        cpu_w=float(cpu_mw) / 1000.0,
                        dram_w=float(dram_mw) / 1000.0 if dram_mw is not None else None,
                    )

                with self._condition:
                    existing = self._readings.get(key)
                    if existing is not None:
                        existing.cpu_power_w = cpu_power_w
                        existing.timestamp_s = max(existing.timestamp_s, timestamp_ms / 1000.0)
                    else:
                        self._readings[key] = PowerReadings(
                            timestamp_s=timestamp_ms / 1000.0,
                            cpu_power_w=cpu_power_w,
                        )
                    self._condition.notify_all()
