"""Stream GPU power readings from remote zeusd instances via SSE.

This module provides `PowerStreamingClient`, a thread-based SSE client
that connects to one or more zeusd TCP endpoints and provides the latest
GPU power readings in a thread-safe manner.

```python
client = PowerStreamingClient(
    endpoints=[("node1", 4938), ("node2", 4938)],
    gpu_indices_by_endpoint={"node1:4938": [0, 1, 2, 3]},
)
client.start()
readings = client.get_power()  # {"node1:4938": PowerReadings(...)}
client.stop()
```
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass
class PowerReadings:
    """Power readings from a single zeusd endpoint.

    Args:
        timestamp_s: Unix timestamp (seconds) of the reading.
        gpu_power_w: Per-GPU power in watts, keyed by GPU index.
    """

    timestamp_s: float = 0.0
    gpu_power_w: dict[int, float] = field(default_factory=dict)


class PowerStreamingClient:
    """Connect to multiple zeusd instances and stream GPU power readings.

    One background thread per endpoint maintains an SSE connection to the
    zeusd `GET /gpu/power/stream` endpoint. The latest power readings are
    stored in a thread-safe dict, accessible via `get_power()`.

    Args:
        endpoints: List of `(host, port)` tuples for zeusd TCP endpoints.
        gpu_indices_by_endpoint: Optional mapping of `"host:port"` to list of
            GPU indices to stream. If a key is missing or the dict is None,
            all GPUs on that endpoint are streamed.
        reconnect_delay_s: Seconds to wait before reconnecting after a
            disconnect.
    """

    def __init__(
        self,
        endpoints: Sequence[tuple[str, int]],
        gpu_indices_by_endpoint: dict[str, list[int]] | None = None,
        reconnect_delay_s: float = 1.0,
    ) -> None:
        """Initialize the client with endpoint addresses and optional GPU filters."""
        self._endpoints = [(h, p) for h, p in endpoints]
        self._gpu_indices = gpu_indices_by_endpoint or {}
        self._reconnect_delay = reconnect_delay_s

        self._lock = threading.Lock()
        self._readings: dict[str, PowerReadings] = {}

        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start background SSE connections to all endpoints."""
        if self._threads:
            raise RuntimeError("PowerStreamingClient is already started")

        self._stop_event.clear()
        for host, port in self._endpoints:
            key = f"{host}:{port}"
            gpu_ids = self._gpu_indices.get(key)
            t = threading.Thread(
                target=self._stream_loop,
                args=(host, port, key, gpu_ids),
                name=f"power-stream-{key}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)
            logger.info("Started power streaming thread for %s", key)

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
            timestamp and per-GPU power in watts.
        """
        with self._lock:
            return {
                k: PowerReadings(
                    timestamp_s=v.timestamp_s,
                    gpu_power_w=dict(v.gpu_power_w),
                )
                for k, v in self._readings.items()
            }

    def _stream_loop(
        self,
        host: str,
        port: int,
        key: str,
        gpu_ids: list[int] | None,
    ) -> None:
        """Background thread: connect to SSE, parse events, update readings."""
        base_url = f"http://{host}:{port}/gpu/power/stream"
        if gpu_ids is not None:
            ids_param = ",".join(str(i) for i in gpu_ids)
            url = f"{base_url}?gpu_ids={ids_param}"
        else:
            url = base_url

        while not self._stop_event.is_set():
            try:
                self._connect_and_stream(url, key)
            except Exception:
                if not self._stop_event.is_set():
                    logger.warning(
                        "SSE connection to %s lost, reconnecting in %.1fs",
                        key,
                        self._reconnect_delay,
                        exc_info=True,
                    )
                    self._stop_event.wait(timeout=self._reconnect_delay)

    def _connect_and_stream(self, url: str, key: str) -> None:
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
                    self._process_event(event_text, key)

    def _process_event(self, event_text: str, key: str) -> None:
        """Parse a single SSE event and update readings."""
        for line in event_text.strip().split("\n"):
            if line.startswith("data: "):
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in SSE event from %s: %s", key, data_str[:100])
                    continue

                power_mw = data.get("power_mw", {})
                timestamp_ms = data.get("timestamp_ms", 0)

                gpu_power_w: dict[int, float] = {}
                for gpu_id_str, mw in power_mw.items():
                    gpu_power_w[int(gpu_id_str)] = float(mw) / 1000.0  # mW -> W

                with self._lock:
                    self._readings[key] = PowerReadings(
                        timestamp_s=timestamp_ms / 1000.0,  # ms -> s
                        gpu_power_w=gpu_power_w,
                    )
