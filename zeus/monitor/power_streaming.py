"""Stream GPU and CPU power readings from remote zeusd instances via SSE.

This module provides `PowerStreamingClient`, a thread-based SSE client
that connects to one or more zeusd TCP endpoints and provides the latest
GPU and CPU power readings in a thread-safe manner.

```python
from zeus.monitor.power_streaming import PowerStreamingClient, ZeusdServerConfig

client = PowerStreamingClient(
    servers=[
        ZeusdServerConfig(
            host="node1", port=4938,
            gpu_indices=[0, 1, 2, 3],
            collect_cpu=True,
        ),
        ZeusdServerConfig(host="node2", port=4938),
    ],
)
readings = client.get_power()  # {"node1:4938": PowerReadings(...)}
client.stop()
```
"""

from __future__ import annotations

import json
import logging
import threading
import typing
from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZeusdServerConfig:
    """Connection configuration for a single zeusd instance.

    Args:
        host: Hostname or IP of the zeusd instance.
        port: TCP port of the zeusd instance.
        gpu_indices: GPU device indices to stream. If None, all GPUs on
            the endpoint are streamed.
        cpu_indices: CPU device indices to stream. If None, all CPUs on
            the endpoint are streamed. Only used when `collect_cpu` is True.
        collect_gpu: Whether to stream GPU power from this server.
        collect_cpu: Whether to stream CPU power from this server.
    """

    host: str
    port: int = 4938
    gpu_indices: list[int] | None = None
    cpu_indices: list[int] | None = None
    collect_gpu: bool = True
    collect_cpu: bool = False

    @property
    def key(self) -> str:
        """Return the `host:port` identifier for this server."""
        return f"{self.host}:{self.port}"


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

    Args:
        servers: List of `ZeusdServerConfig` specifying zeusd endpoints
            and which GPUs/CPUs to stream from each.
        reconnect_delay_s: Seconds to wait before reconnecting after a
            disconnect.
    """

    def __init__(
        self,
        servers: Sequence[ZeusdServerConfig],
        reconnect_delay_s: float = 1.0,
    ) -> None:
        """Initialize the client and start background SSE connections."""
        self._servers = list(servers)
        self._reconnect_delay = reconnect_delay_s

        self._lock = threading.Lock()
        self._readings: dict[str, PowerReadings] = {}

        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

        for server in self._servers:
            if server.collect_gpu:
                t = threading.Thread(
                    target=self._gpu_stream_loop,
                    args=(server,),
                    name=f"gpu-power-stream-{server.key}",
                    daemon=True,
                )
                t.start()
                self._threads.append(t)
                logger.info("Started GPU power streaming thread for %s", server.key)

            if server.collect_cpu:
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
                    logger.warning(
                        "CPU power collection requested for %s but RAPL is not "
                        "available on that server; skipping CPU streaming",
                        server.key,
                    )

    def stop(self) -> None:
        """Stop all background connections."""
        self._stop_event.set()
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

    def _check_cpu_available(self, server: ZeusdServerConfig) -> bool:
        """Probe the one-shot CPU power endpoint to check RAPL availability."""
        url = f"http://{server.host}:{server.port}/cpu/power"
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()
                power_mw = data.get("power_mw", {})
                return len(power_mw) > 0
        except (httpx.RequestError, httpx.HTTPStatusError):
            logger.warning("Failed to probe CPU power endpoint on %s", server.key, exc_info=True)
            return False

    def _gpu_stream_loop(self, server: ZeusdServerConfig) -> None:
        """Background thread: stream GPU power from a single server."""
        base_url = f"http://{server.host}:{server.port}/gpu/power/stream"
        if server.gpu_indices is not None:
            ids_param = ",".join(str(i) for i in server.gpu_indices)
            url = f"{base_url}?gpu_ids={ids_param}"
        else:
            url = base_url
        self._stream_loop(url, server.key, self._process_gpu_event, "GPU")

    def _cpu_stream_loop(self, server: ZeusdServerConfig) -> None:
        """Background thread: stream CPU power from a single server."""
        base_url = f"http://{server.host}:{server.port}/cpu/power/stream"
        if server.cpu_indices is not None:
            ids_param = ",".join(str(i) for i in server.cpu_indices)
            url = f"{base_url}?cpu_ids={ids_param}"
        else:
            url = base_url
        self._stream_loop(url, server.key, self._process_cpu_event, "CPU")

    def _stream_loop(
        self,
        url: str,
        key: str,
        process_fn: typing.Callable[[str, str], None],
        label: str,
    ) -> None:
        """Shared reconnect loop for SSE streams."""
        while not self._stop_event.is_set():
            try:
                self._connect_and_stream(url, key, process_fn)
            except httpx.HTTPStatusError:
                if not self._stop_event.is_set():
                    logger.warning(
                        "%s SSE connection to %s rejected, reconnecting in %.1fs",
                        label,
                        key,
                        self._reconnect_delay,
                        exc_info=True,
                    )
                    self._stop_event.wait(timeout=self._reconnect_delay)
            except httpx.RequestError:
                if not self._stop_event.is_set():
                    logger.warning(
                        "%s SSE connection to %s lost, reconnecting in %.1fs",
                        label,
                        key,
                        self._reconnect_delay,
                        exc_info=True,
                    )
                    self._stop_event.wait(timeout=self._reconnect_delay)

    def _connect_and_stream(
        self,
        url: str,
        key: str,
        process_fn: typing.Callable[[str, str], None],
    ) -> None:
        """Open an SSE connection and process events until disconnected."""
        logger.info("Connecting to SSE at %s", url)
        with httpx.Client(timeout=None) as client, client.stream("GET", url) as response:
            response.raise_for_status()
            logger.info("SSE connected to %s", url)
            buffer = ""
            for chunk in response.iter_text():
                if self._stop_event.is_set():
                    return
                buffer += chunk
                while "\n\n" in buffer:
                    event_text, buffer = buffer.split("\n\n", 1)
                    process_fn(event_text, key)

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

                with self._lock:
                    existing = self._readings.get(key)
                    if existing is not None:
                        existing.gpu_power_w = gpu_power_w
                        existing.timestamp_s = max(existing.timestamp_s, timestamp_ms / 1000.0)
                    else:
                        self._readings[key] = PowerReadings(
                            timestamp_s=timestamp_ms / 1000.0,
                            gpu_power_w=gpu_power_w,
                        )

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

                with self._lock:
                    existing = self._readings.get(key)
                    if existing is not None:
                        existing.cpu_power_w = cpu_power_w
                        existing.timestamp_s = max(existing.timestamp_s, timestamp_ms / 1000.0)
                    else:
                        self._readings[key] = PowerReadings(
                            timestamp_s=timestamp_ms / 1000.0,
                            cpu_power_w=cpu_power_w,
                        )
