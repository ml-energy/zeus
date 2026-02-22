"""Zeusd client library.

Provides `ZeusdConfig` and `ZeusdClient`, the entry points for
communicating with a Zeusd daemon.  Handles connection, discovery,
authentication, and exposes typed methods for every Zeusd endpoint.

Typical usage:

```python
from zeus.utils.zeusd import ZeusdConfig, ZeusdClient

client = ZeusdClient(ZeusdConfig.uds(socket_path="/var/run/zeusd.sock"))
print(client.gpu_ids)       # [0, 1, 2, 3]
print(client.can_read_gpu)  # True

snapshot = client.get_gpu_power()
print(snapshot.power_mw)    # {0: 75000, 1: 120000, ...}
```
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

from zeus.exception import ZeusBaseError

logger = logging.getLogger(__name__)


class ZeusdConnectionError(ZeusBaseError):
    """Cannot reach the Zeusd daemon."""


class ZeusdAuthError(ZeusBaseError):
    """Authentication or authorization failure."""


class ZeusdCapabilityError(ZeusBaseError):
    """Requested capabilities exceed what the daemon offers."""


@dataclass(frozen=True)
class GpuPowerSnapshot:
    """Instantaneous GPU power readings from the daemon.

    Attributes:
        timestamp_ms: Daemon-side Unix timestamp in milliseconds.
        power_mw: Mapping of GPU index to power draw in milliwatts.
    """

    timestamp_ms: int
    power_mw: dict[int, int]


@dataclass(frozen=True)
class CpuDramPower:
    """Power reading for a single CPU package.

    Attributes:
        cpu_mw: CPU package power in milliwatts.
        dram_mw: DRAM power in milliwatts, or None if unavailable.
    """

    cpu_mw: int
    dram_mw: int | None


@dataclass(frozen=True)
class CpuPowerSnapshot:
    """Instantaneous CPU power readings from the daemon.

    Attributes:
        timestamp_ms: Daemon-side Unix timestamp in milliseconds.
        power_mw: Mapping of CPU index to power readings.
    """

    timestamp_ms: int
    power_mw: dict[int, CpuDramPower]


@dataclass(frozen=True)
class CpuEnergyResult:
    """Cumulative energy for a single CPU package.

    Attributes:
        cpu_energy_uj: CPU package energy in microjoules, or None.
        dram_energy_uj: DRAM energy in microjoules, or None.
    """

    cpu_energy_uj: int | None
    dram_energy_uj: int | None


@dataclass(frozen=True)
class ZeusdConfig:
    """Connection configuration for a Zeusd daemon.

    Use the classmethods `tcp`, `uds`, or `from_env` to construct.

    Attributes:
        host: Hostname or IP (TCP mode). None for UDS.
        port: TCP port (default 4938). Ignored for UDS.
        socket_path: Unix domain socket path (UDS mode). None for TCP.
        token: JWT token. Falls back to `ZEUSD_TOKEN` env var.
        gpu_indices: GPU indices to stream (for `PowerStreamingClient`).
            None means all, empty list means skip. Ignored by `ZeusdClient`.
        cpu_indices: CPU indices to stream (for `PowerStreamingClient`).
            None means all, empty list means skip. Ignored by `ZeusdClient`.
    """

    host: str | None = None
    port: int = 4938
    socket_path: str | None = None
    token: str | None = None
    gpu_indices: list[int] | None = None
    cpu_indices: list[int] | None = None

    @classmethod
    def tcp(
        cls,
        host: str,
        port: int = 4938,
        *,
        token: str | None = None,
        gpu_indices: list[int] | None = None,
        cpu_indices: list[int] | None = None,
    ) -> ZeusdConfig:
        """Create a TCP connection config.

        Args:
            host: Hostname or IP of the Zeusd instance.
            port: TCP port (default 4938).
            token: JWT token. Falls back to `ZEUSD_TOKEN` env var.
            gpu_indices: GPU indices to stream (for `PowerStreamingClient`).
            cpu_indices: CPU indices to stream (for `PowerStreamingClient`).
        """
        return cls(
            host=host,
            port=port,
            token=token,
            gpu_indices=gpu_indices,
            cpu_indices=cpu_indices,
        )

    @classmethod
    def uds(
        cls,
        socket_path: str,
        *,
        token: str | None = None,
        gpu_indices: list[int] | None = None,
        cpu_indices: list[int] | None = None,
    ) -> ZeusdConfig:
        """Create a Unix domain socket connection config.

        Args:
            socket_path: Path to the Zeusd Unix domain socket.
            token: JWT token. Falls back to `ZEUSD_TOKEN` env var.
            gpu_indices: GPU indices to stream (for `PowerStreamingClient`).
            cpu_indices: CPU indices to stream (for `PowerStreamingClient`).
        """
        return cls(
            socket_path=socket_path,
            token=token,
            gpu_indices=gpu_indices,
            cpu_indices=cpu_indices,
        )

    @classmethod
    def from_env(cls) -> ZeusdConfig | None:
        """Create from environment variables.

        Tries `ZEUSD_SOCK_PATH` (UDS) first, then `ZEUSD_HOST_PORT` (TCP).
        `ZEUSD_HOST_PORT` accepts `host:port` or just `host` (defaults to
        port 4938). `ZEUSD_TOKEN` is read for JWT authentication.

        Returns None if neither env var is set.
        """
        token = os.environ.get("ZEUSD_TOKEN")
        sock = os.environ.get("ZEUSD_SOCK_PATH")
        if sock is not None:
            return cls.uds(socket_path=sock, token=token)
        host_port = os.environ.get("ZEUSD_HOST_PORT")
        if host_port is not None:
            if ":" in host_port:
                host, port_str = host_port.rsplit(":", 1)
                port = int(port_str)
            else:
                host = host_port
                port = 4938
            return cls.tcp(host=host, port=port, token=token)
        return None

    @property
    def _is_uds(self) -> bool:
        return self.socket_path is not None

    def make_client(self) -> httpx.Client:
        """Create an httpx.Client with the appropriate transport and auth."""
        headers = self._auth_headers()
        if self._is_uds:
            transport = httpx.HTTPTransport(uds=self.socket_path)
            return httpx.Client(transport=transport, headers=headers)
        return httpx.Client(headers=headers)

    def url(self, path: str) -> str:
        """Build the full URL for the given path."""
        if self._is_uds:
            return f"http://localhost{path}"
        return f"http://{self.host}:{self.port}{path}"

    @property
    def endpoint(self) -> str:
        """Human-readable identifier for this connection."""
        if self._is_uds:
            return self.socket_path  # type: ignore[return-value]
        return f"{self.host}:{self.port}"

    def _auth_headers(self) -> dict[str, str]:
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}


class ZeusdClient:
    """Authenticated client for a Zeusd daemon.

    Handles connection, service discovery, and JWT authentication in
    one place.  Provides typed methods for every Zeusd endpoint.

    Args:
        config: Connection configuration.  If None, tries environment
            variables: `ZEUSD_SOCK_PATH` (UDS) first, then
            `ZEUSD_HOST_PORT` (TCP).

    Raises:
        ZeusdConnectionError: If the daemon is unreachable.
    """

    def __init__(self, config: ZeusdConfig | None = None) -> None:
        """Initialize the client, run discovery, and attempt authentication."""
        if config is None:
            config = ZeusdConfig.from_env()
            if config is None:
                raise ZeusdConnectionError(
                    "No Zeusd connection configured. Set ZEUSD_SOCK_PATH or ZEUSD_HOST_PORT, or pass a ZeusdConfig."
                )
        self._config = config
        self._client = config.make_client()

        try:
            resp = self._client.get(config.url("/discover"))
        except httpx.RequestError as exc:
            raise ZeusdConnectionError(f"Cannot reach Zeusd at {config.endpoint}: {exc}") from exc
        if resp.status_code != 200:
            raise ZeusdConnectionError(
                f"Zeusd at {config.endpoint} returned HTTP {resp.status_code} on /discover: {resp.text}"
            )
        data = resp.json()
        self._gpu_ids: list[int] = data.get("gpu_ids", [])
        self._cpu_ids: list[int] = data.get("cpu_ids", [])
        self._dram_available: list[bool] = data.get("dram_available", [])
        self._enabled_api_groups: set[str] = set(data.get("enabled_api_groups", []))
        self._auth_required: bool = data.get("auth_required", False)

        self._auth_error: str | None = None
        self._granted_scopes: frozenset[str] = frozenset()
        self._whoami_sub: str | None = None
        self._whoami_exp: int | None = None
        if self._auth_required:
            if not config.token:
                self._auth_error = (
                    f"Zeusd at {config.endpoint} requires authentication but "
                    "no token was provided. Set the ZEUSD_TOKEN environment "
                    "variable or pass token= in the config."
                )
            else:
                whoami_resp = self._client.get(config.url("/auth/whoami"))
                if whoami_resp.status_code == 401:
                    self._auth_error = f"Token rejected by Zeusd at {config.endpoint}: {whoami_resp.text}"
                elif whoami_resp.status_code != 200:
                    self._auth_error = (
                        f"Unexpected response from /auth/whoami at "
                        f"{config.endpoint} (HTTP {whoami_resp.status_code}): "
                        f"{whoami_resp.text}"
                    )
                else:
                    whoami = whoami_resp.json()
                    self._granted_scopes = frozenset(whoami.get("scopes", []))
                    self._whoami_sub = whoami.get("sub")
                    self._whoami_exp = whoami.get("exp")
                    logger.info(
                        "Authenticated with Zeusd at %s as user '%s' (scopes: %s)",
                        config.endpoint,
                        self._whoami_sub,
                        sorted(self._granted_scopes),
                    )
            if self._auth_error:
                logger.warning("Auth issue with Zeusd at %s: %s", config.endpoint, self._auth_error)

    @property
    def endpoint(self) -> str:
        """Human-readable identifier for this connection."""
        return self._config.endpoint

    @property
    def gpu_ids(self) -> list[int]:
        """GPU device indices available on this daemon."""
        return list(self._gpu_ids)

    @property
    def cpu_ids(self) -> list[int]:
        """CPU device indices available on this daemon."""
        return list(self._cpu_ids)

    @property
    def dram_available(self) -> list[bool]:
        """Per-CPU DRAM energy availability, aligned with `cpu_ids`."""
        return list(self._dram_available)

    @property
    def auth_required(self) -> bool:
        """Whether this daemon requires JWT authentication."""
        return self._auth_required

    @property
    def auth_error(self) -> str | None:
        """Auth error message, or None if auth succeeded or is not required."""
        return self._auth_error

    @property
    def granted_scopes(self) -> frozenset[str]:
        """Scopes granted by the current token (empty if auth is off or failed)."""
        return self._granted_scopes

    def _can(self, api_group: str, scope: str) -> bool:
        if api_group not in self._enabled_api_groups:
            return False
        return not (self._auth_required and scope not in self._granted_scopes)

    @property
    def can_read_gpu(self) -> bool:
        """Whether GPU read endpoints are accessible."""
        return self._can("gpu-read", "gpu-read")

    @property
    def can_control_gpu(self) -> bool:
        """Whether GPU control endpoints are accessible."""
        return self._can("gpu-control", "gpu-control")

    @property
    def can_read_cpu(self) -> bool:
        """Whether CPU read endpoints are accessible."""
        return self._can("cpu-read", "cpu-read")

    def get_gpu_energy(self, gpu_ids: list[int]) -> dict[int, int]:
        """Get cumulative energy consumption per GPU.

        Args:
            gpu_ids: GPU indices to query.

        Returns:
            Mapping of GPU index to cumulative energy in millijoules.
        """
        resp = self._client.get(
            self._config.url("/gpu/get_cumulative_energy"),
            params={"gpu_ids": ",".join(str(i) for i in gpu_ids)},
        )
        self._check(resp, "get_gpu_energy")
        data = resp.json()
        return {int(k): v["energy_mj"] for k, v in data.items()}

    def get_gpu_power(self, gpu_ids: list[int] | None = None) -> GpuPowerSnapshot:
        """Get instantaneous GPU power readings.

        Args:
            gpu_ids: GPU indices to query.  None means all.

        Returns:
            Snapshot with timestamp and per-GPU power in milliwatts.
        """
        params: dict[str, str] = {}
        if gpu_ids is not None:
            params["gpu_ids"] = ",".join(str(i) for i in gpu_ids)
        resp = self._client.get(self._config.url("/gpu/get_power"), params=params)
        self._check(resp, "get_gpu_power")
        data = resp.json()
        return GpuPowerSnapshot(
            timestamp_ms=data["timestamp_ms"],
            power_mw={int(k): v for k, v in data["power_mw"].items()},
        )

    def set_power_limit(self, gpu_ids: list[int], power_limit_mw: int, block: bool = True) -> None:
        """Set the power management limit for the given GPUs."""
        resp = self._client.post(
            self._config.url("/gpu/set_power_limit"),
            params={
                "gpu_ids": ",".join(str(i) for i in gpu_ids),
                "power_limit_mw": str(power_limit_mw),
                "block": "true" if block else "false",
            },
        )
        self._check(resp, "set_power_limit")

    def set_persistence_mode(self, gpu_ids: list[int], enabled: bool, block: bool = True) -> None:
        """Set persistence mode for the given GPUs."""
        resp = self._client.post(
            self._config.url("/gpu/set_persistence_mode"),
            params={
                "gpu_ids": ",".join(str(i) for i in gpu_ids),
                "enabled": "true" if enabled else "false",
                "block": "true" if block else "false",
            },
        )
        self._check(resp, "set_persistence_mode")

    def set_gpu_locked_clocks(
        self,
        gpu_ids: list[int],
        min_clock_mhz: int,
        max_clock_mhz: int,
        block: bool = True,
    ) -> None:
        """Lock the GPU clock to a specified range (MHz)."""
        resp = self._client.post(
            self._config.url("/gpu/set_gpu_locked_clocks"),
            params={
                "gpu_ids": ",".join(str(i) for i in gpu_ids),
                "min_clock_mhz": str(min_clock_mhz),
                "max_clock_mhz": str(max_clock_mhz),
                "block": "true" if block else "false",
            },
        )
        self._check(resp, "set_gpu_locked_clocks")

    def reset_gpu_locked_clocks(self, gpu_ids: list[int], block: bool = True) -> None:
        """Reset locked GPU clocks to the default."""
        resp = self._client.post(
            self._config.url("/gpu/reset_gpu_locked_clocks"),
            params={
                "gpu_ids": ",".join(str(i) for i in gpu_ids),
                "block": "true" if block else "false",
            },
        )
        self._check(resp, "reset_gpu_locked_clocks")

    def set_mem_locked_clocks(
        self,
        gpu_ids: list[int],
        min_clock_mhz: int,
        max_clock_mhz: int,
        block: bool = True,
    ) -> None:
        """Lock the memory clock to a specified range (MHz)."""
        resp = self._client.post(
            self._config.url("/gpu/set_mem_locked_clocks"),
            params={
                "gpu_ids": ",".join(str(i) for i in gpu_ids),
                "min_clock_mhz": str(min_clock_mhz),
                "max_clock_mhz": str(max_clock_mhz),
                "block": "true" if block else "false",
            },
        )
        self._check(resp, "set_mem_locked_clocks")

    def reset_mem_locked_clocks(self, gpu_ids: list[int], block: bool = True) -> None:
        """Reset locked memory clocks to the default."""
        resp = self._client.post(
            self._config.url("/gpu/reset_mem_locked_clocks"),
            params={
                "gpu_ids": ",".join(str(i) for i in gpu_ids),
                "block": "true" if block else "false",
            },
        )
        self._check(resp, "reset_mem_locked_clocks")

    def get_cpu_energy(
        self,
        cpu_ids: list[int],
        cpu: bool = True,
        dram: bool = True,
    ) -> dict[int, CpuEnergyResult]:
        """Get cumulative energy consumption per CPU.

        Args:
            cpu_ids: CPU indices to query.
            cpu: Whether to include CPU package energy.
            dram: Whether to include DRAM energy.

        Returns:
            Mapping of CPU index to energy results.
        """
        resp = self._client.get(
            self._config.url("/cpu/get_cumulative_energy"),
            params={
                "cpu_ids": ",".join(str(i) for i in cpu_ids),
                "cpu": "true" if cpu else "false",
                "dram": "true" if dram else "false",
            },
        )
        self._check(resp, "get_cpu_energy")
        data = resp.json()
        return {
            int(k): CpuEnergyResult(
                cpu_energy_uj=v.get("cpu_energy_uj"),
                dram_energy_uj=v.get("dram_energy_uj"),
            )
            for k, v in data.items()
        }

    def get_cpu_power(self, cpu_ids: list[int] | None = None) -> CpuPowerSnapshot:
        """Get instantaneous CPU power readings.

        Args:
            cpu_ids: CPU indices to query.  None means all.

        Returns:
            Snapshot with timestamp and per-CPU power in milliwatts.
        """
        params: dict[str, str] = {}
        if cpu_ids is not None:
            params["cpu_ids"] = ",".join(str(i) for i in cpu_ids)
        resp = self._client.get(self._config.url("/cpu/get_power"), params=params)
        self._check(resp, "get_cpu_power")
        data = resp.json()
        return CpuPowerSnapshot(
            timestamp_ms=data["timestamp_ms"],
            power_mw={
                int(k): CpuDramPower(cpu_mw=v["cpu_mw"], dram_mw=v.get("dram_mw")) for k, v in data["power_mw"].items()
            },
        )

    def get_time(self) -> float:
        """Get daemon timestamp in seconds."""
        resp = self._client.get(self._config.url("/time"))
        self._check(resp, "get_time")
        return resp.json()["timestamp_ms"] / 1000.0

    def make_client(self) -> httpx.Client:
        """Create a new httpx.Client with this client's transport and auth.

        Used by `PowerStreamingClient` for SSE streaming connections
        where a dedicated, long-lived httpx.Client is needed.
        """
        return self._config.make_client()

    def url(self, path: str) -> str:
        """Build the full URL for the given path.

        Used together with `make_client()` for streaming URLs.
        """
        return self._config.url(path)

    @staticmethod
    def _check(resp: httpx.Response, operation: str) -> None:
        """Raise ZeusdError if the response is not 200."""
        if resp.status_code != 200:
            # Import here to avoid circular import at module level.
            from zeus.device.exception import ZeusdError

            raise ZeusdError(f"Failed to {operation}: {resp.text}")


def require_capabilities(
    client: ZeusdClient,
    *,
    read_gpu: bool = False,
    control_gpu: bool = False,
    read_cpu: bool = False,
    gpu_ids: list[int] | None = None,
    cpu_ids: list[int] | None = None,
) -> None:
    """Fail-fast validation that the daemon supports what the caller needs.

    Checks that the required API groups are enabled, the required scopes
    are granted by the token, and that the requested device IDs are
    available on the daemon.

    Args:
        client: The ZeusdClient to validate against.
        read_gpu: Require the gpu-read capability.
        control_gpu: Require the gpu-control capability.
        read_cpu: Require the cpu-read capability.
        gpu_ids: GPU indices that must be available.
        cpu_ids: CPU indices that must be available.

    Raises:
        ZeusdCapabilityError: If any requirement is not met.
    """
    if client.auth_error:
        raise ZeusdAuthError(client.auth_error)

    errors: list[str] = []

    if read_gpu and not client.can_read_gpu:
        errors.append(_capability_reason(client, "gpu-read"))
    if control_gpu and not client.can_control_gpu:
        errors.append(_capability_reason(client, "gpu-control"))
    if read_cpu and not client.can_read_cpu:
        errors.append(_capability_reason(client, "cpu-read"))

    if gpu_ids is not None:
        available = set(client.gpu_ids)
        missing = set(gpu_ids) - available
        if missing:
            errors.append(f"GPU indices {sorted(missing)} not available (available: {sorted(available)})")

    if cpu_ids is not None:
        available = set(client.cpu_ids)
        missing = set(cpu_ids) - available
        if missing:
            errors.append(f"CPU indices {sorted(missing)} not available (available: {sorted(available)})")

    if errors:
        raise ZeusdCapabilityError(f"Zeusd at {client.endpoint}: " + "; ".join(errors))


def _capability_reason(client: ZeusdClient, scope: str) -> str:
    """Build a human-readable reason why a capability is unavailable."""
    if scope not in client._enabled_api_groups:
        return f"API group '{scope}' is not enabled on this server"
    if client.auth_required and scope not in client.granted_scopes:
        return f"Token lacks required scope '{scope}' (granted: {sorted(client.granted_scopes)})"
    return f"'{scope}' is not available"
