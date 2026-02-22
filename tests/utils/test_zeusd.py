"""Tests for zeus.utils.zeusd: ZeusdConfig, ZeusdClient, require_capabilities."""

from __future__ import annotations

import httpx
import pytest

from zeus.device.exception import ZeusdError
from zeus.utils.zeusd import (
    CpuDramPower,
    CpuEnergyResult,
    CpuPowerSnapshot,
    GpuPowerSnapshot,
    ZeusdAuthError,
    ZeusdCapabilityError,
    ZeusdClient,
    ZeusdConfig,
    ZeusdConnectionError,
    require_capabilities,
)


# ---------------------------------------------------------------------------
# Mock server fixture
# ---------------------------------------------------------------------------


class MockZeusdServer:
    """Holds the config and captured requests from a mock Zeusd server."""

    def __init__(self, config: ZeusdConfig, requests: list[httpx.Request]):
        self.config = config
        self.requests = requests

    def last_request(self) -> httpx.Request:
        return self.requests[-1]

    def last_params(self) -> dict[str, str]:
        return dict(self.last_request().url.params)


@pytest.fixture()
def mock_zeusd(monkeypatch):
    """Factory fixture: configure a mock Zeusd daemon and return a MockZeusdServer.

    The fixture patches ``ZeusdConfig.make_client`` so that all HTTP traffic
    goes through an ``httpx.MockTransport``.  Server behaviour is controlled
    entirely via keyword arguments.
    """

    def _configure(
        *,
        gpu_ids: tuple[int, ...] = (0, 1, 2, 3),
        cpu_ids: tuple[int, ...] = (0, 1),
        dram_available: tuple[bool, ...] = (True, False),
        enabled_api_groups: tuple[str, ...] = (
            "gpu-control",
            "gpu-read",
            "cpu-read",
        ),
        auth_required: bool = False,
        whoami_sub: str = "alice",
        whoami_scopes: tuple[str, ...] = (
            "gpu-control",
            "gpu-read",
            "cpu-read",
        ),
        whoami_exp: int | None = None,
        whoami_status: int = 200,
        token: str | None = None,
        endpoint_errors: dict[str, int] | None = None,
    ) -> MockZeusdServer:
        _endpoint_errors = endpoint_errors or {}

        discover_body = {
            "gpu_ids": list(gpu_ids),
            "cpu_ids": list(cpu_ids),
            "dram_available": list(dram_available),
            "enabled_api_groups": list(enabled_api_groups),
            "auth_required": auth_required,
        }

        whoami_body: dict = {"sub": whoami_sub, "scopes": list(whoami_scopes)}
        if whoami_exp is not None:
            whoami_body["exp"] = whoami_exp

        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            path = request.url.path

            if path in _endpoint_errors:
                return httpx.Response(_endpoint_errors[path], text=f"Mocked error for {path}")

            if path == "/discover":
                return httpx.Response(200, json=discover_body)

            if path == "/auth/whoami":
                if whoami_status != 200:
                    body = {"error": "Token has expired."} if whoami_status == 401 else {"error": "Server error"}
                    return httpx.Response(whoami_status, json=body)
                return httpx.Response(200, json=whoami_body)

            if path == "/time":
                return httpx.Response(200, json={"timestamp_ms": 1700000000000})

            if path == "/gpu/get_power":
                return httpx.Response(
                    200,
                    json={
                        "timestamp_ms": 1700000000000,
                        "power_mw": {str(i): 75000 + i * 1000 for i in gpu_ids},
                    },
                )

            if path == "/gpu/get_cumulative_energy":
                return httpx.Response(
                    200,
                    json={str(i): {"energy_mj": 100000 + i * 10000} for i in gpu_ids},
                )

            if path == "/cpu/get_power":
                return httpx.Response(
                    200,
                    json={
                        "timestamp_ms": 1700000000000,
                        "power_mw": {
                            str(i): {
                                "cpu_mw": 50000 + i * 5000,
                                "dram_mw": 10000 if dram_available[j] else None,
                            }
                            for j, i in enumerate(cpu_ids)
                        },
                    },
                )

            if path == "/cpu/get_cumulative_energy":
                return httpx.Response(
                    200,
                    json={
                        str(i): {
                            "cpu_energy_uj": 500000 + i * 50000,
                            "dram_energy_uj": 100000 if dram_available[j] else None,
                        }
                        for j, i in enumerate(cpu_ids)
                    },
                )

            # GPU control endpoints all return 200.
            if path.startswith("/gpu/"):
                return httpx.Response(200, text="OK")

            return httpx.Response(404, text=f"Not found: {path}")

        def patched_make_client(self_config):
            headers = self_config._auth_headers()
            return httpx.Client(transport=httpx.MockTransport(handler), headers=headers)

        monkeypatch.setattr(ZeusdConfig, "make_client", patched_make_client)
        config = ZeusdConfig.tcp(host="testhost", port=4938, token=token)
        return MockZeusdServer(config=config, requests=captured)

    return _configure


# ---------------------------------------------------------------------------
# ZeusdConfig
# ---------------------------------------------------------------------------


class TestZeusdConfig:
    def test_tcp(self):
        c = ZeusdConfig.tcp("node1", 5000, token="tok", gpu_indices=[0], cpu_indices=[1])
        assert c.host_port == "node1:5000"
        assert c.socket_path is None
        assert c.token == "tok"
        assert c.gpu_indices == [0]
        assert c.cpu_indices == [1]

    def test_tcp_defaults(self):
        c = ZeusdConfig.tcp("node1", 4938)
        assert c.host_port == "node1:4938"
        assert c.token is None
        assert c.gpu_indices is None
        assert c.cpu_indices is None

    def test_uds(self):
        c = ZeusdConfig.uds("/var/run/zeusd.sock", token="tok")
        assert c.socket_path == "/var/run/zeusd.sock"
        assert c.host_port is None
        assert c.token == "tok"

    def test_from_env_uds(self, monkeypatch):
        monkeypatch.setenv("ZEUSD_SOCK_PATH", "/tmp/test.sock")
        monkeypatch.delenv("ZEUSD_HOST_PORT", raising=False)
        c = ZeusdConfig.from_env()
        assert c is not None
        assert c.socket_path == "/tmp/test.sock"

    def test_from_env_tcp_host_port(self, monkeypatch):
        monkeypatch.delenv("ZEUSD_SOCK_PATH", raising=False)
        monkeypatch.setenv("ZEUSD_HOST_PORT", "node1:5000")
        c = ZeusdConfig.from_env()
        assert c is not None
        assert c.host_port == "node1:5000"

    def test_from_env_tcp_host_only(self, monkeypatch):
        monkeypatch.delenv("ZEUSD_SOCK_PATH", raising=False)
        monkeypatch.setenv("ZEUSD_HOST_PORT", "node1")
        c = ZeusdConfig.from_env()
        assert c is not None
        assert c.host_port == "node1"

    def test_from_env_token(self, monkeypatch):
        monkeypatch.setenv("ZEUSD_SOCK_PATH", "/tmp/test.sock")
        monkeypatch.setenv("ZEUSD_TOKEN", "mytoken")
        c = ZeusdConfig.from_env()
        assert c is not None
        assert c.token == "mytoken"

    def test_from_env_none(self, monkeypatch):
        monkeypatch.delenv("ZEUSD_SOCK_PATH", raising=False)
        monkeypatch.delenv("ZEUSD_HOST_PORT", raising=False)
        assert ZeusdConfig.from_env() is None

    def test_from_env_uds_takes_priority(self, monkeypatch):
        monkeypatch.setenv("ZEUSD_SOCK_PATH", "/tmp/test.sock")
        monkeypatch.setenv("ZEUSD_HOST_PORT", "node1:5000")
        c = ZeusdConfig.from_env()
        assert c is not None
        assert c.socket_path == "/tmp/test.sock"
        assert c.host_port is None

    def test_url_tcp(self):
        c = ZeusdConfig.tcp("node1", 5000)
        assert c.url("/discover") == "http://node1:5000/discover"

    def test_url_uds(self):
        c = ZeusdConfig.uds("/var/run/zeusd.sock")
        assert c.url("/discover") == "http://localhost/discover"

    def test_endpoint_tcp(self):
        assert ZeusdConfig.tcp("node1", 5000).endpoint == "node1:5000"

    def test_endpoint_uds(self):
        assert ZeusdConfig.uds("/var/run/zeusd.sock").endpoint == "/var/run/zeusd.sock"


# ---------------------------------------------------------------------------
# ZeusdClient initialisation
# ---------------------------------------------------------------------------


class TestZeusdClientInit:
    def test_no_config_no_env(self, monkeypatch):
        monkeypatch.delenv("ZEUSD_SOCK_PATH", raising=False)
        monkeypatch.delenv("ZEUSD_HOST_PORT", raising=False)
        with pytest.raises(ZeusdConnectionError, match="No Zeusd connection"):
            ZeusdClient()

    def test_connection_error(self, monkeypatch):
        def failing_make_client(self_config):
            def handler(request: httpx.Request) -> httpx.Response:
                raise httpx.ConnectError("Connection refused")

            return httpx.Client(transport=httpx.MockTransport(handler))

        monkeypatch.setattr(ZeusdConfig, "make_client", failing_make_client)
        config = ZeusdConfig.tcp("unreachable", 4938)
        with pytest.raises(ZeusdConnectionError, match="Cannot reach Zeusd"):
            ZeusdClient(config)

    def test_discover_non_200(self, mock_zeusd):
        server = mock_zeusd(endpoint_errors={"/discover": 503})
        with pytest.raises(ZeusdConnectionError, match="HTTP 503"):
            ZeusdClient(server.config)

    def test_discover_parses_response(self, mock_zeusd):
        server = mock_zeusd(
            gpu_ids=(0, 1),
            cpu_ids=(2,),
            dram_available=(True,),
            enabled_api_groups=("gpu-read",),
        )
        client = ZeusdClient(server.config)
        assert client.gpu_ids == [0, 1]
        assert client.cpu_ids == [2]
        assert client.dram_available == [True]
        assert not client.auth_required

    def test_no_auth_required(self, mock_zeusd):
        server = mock_zeusd(auth_required=False)
        client = ZeusdClient(server.config)
        assert not client.auth_required
        assert client.auth_error is None
        assert client.granted_scopes == frozenset()
        assert client._whoami_sub is None
        assert client._whoami_exp is None

    def test_auth_no_token_stores_error(self, mock_zeusd):
        server = mock_zeusd(auth_required=True, token=None)
        client = ZeusdClient(server.config)
        assert client.auth_required
        assert client.auth_error is not None
        assert "no token was provided" in client.auth_error
        # Discovery results are still available.
        assert client.gpu_ids == [0, 1, 2, 3]

    def test_auth_expired_token_stores_error(self, mock_zeusd):
        server = mock_zeusd(auth_required=True, token="expired", whoami_status=401)
        client = ZeusdClient(server.config)
        assert client.auth_error is not None
        assert "Token rejected" in client.auth_error
        assert client.gpu_ids == [0, 1, 2, 3]

    def test_auth_whoami_server_error_stores_error(self, mock_zeusd):
        server = mock_zeusd(auth_required=True, token="valid", whoami_status=500)
        client = ZeusdClient(server.config)
        assert client.auth_error is not None
        assert "HTTP 500" in client.auth_error

    def test_auth_valid_token(self, mock_zeusd):
        server = mock_zeusd(
            auth_required=True,
            token="valid",
            whoami_sub="bob",
            whoami_scopes=("gpu-read", "cpu-read"),
            whoami_exp=9999999999,
        )
        client = ZeusdClient(server.config)
        assert client.auth_error is None
        assert client._whoami_sub == "bob"
        assert client.granted_scopes == frozenset({"gpu-read", "cpu-read"})
        assert client._whoami_exp == 9999999999

    def test_auth_valid_token_no_expiry(self, mock_zeusd):
        server = mock_zeusd(
            auth_required=True,
            token="valid",
            whoami_sub="alice",
            whoami_scopes=("gpu-read",),
        )
        client = ZeusdClient(server.config)
        assert client.auth_error is None
        assert client._whoami_exp is None

    def test_from_env_fallback(self, mock_zeusd, monkeypatch):
        mock_zeusd(gpu_ids=(5,), cpu_ids=(), dram_available=())
        monkeypatch.delenv("ZEUSD_SOCK_PATH", raising=False)
        monkeypatch.setenv("ZEUSD_HOST_PORT", "testhost:4938")
        client = ZeusdClient()
        assert client.gpu_ids == [5]


# ---------------------------------------------------------------------------
# ZeusdClient capability checks
# ---------------------------------------------------------------------------


class TestZeusdClientCapabilities:
    @pytest.mark.parametrize(
        "groups, can_read, can_control, can_cpu",
        [
            (("gpu-read",), True, False, False),
            (("gpu-control",), False, True, False),
            (("cpu-read",), False, False, True),
            (("gpu-read", "gpu-control"), True, True, False),
            (("gpu-read", "cpu-read"), True, False, True),
            (("gpu-control", "gpu-read", "cpu-read"), True, True, True),
            ((), False, False, False),
        ],
    )
    def test_no_auth(self, mock_zeusd, groups, can_read, can_control, can_cpu):
        server = mock_zeusd(enabled_api_groups=groups, auth_required=False)
        client = ZeusdClient(server.config)
        assert client.can_read_gpu == can_read
        assert client.can_control_gpu == can_control
        assert client.can_read_cpu == can_cpu

    @pytest.mark.parametrize(
        "scopes, can_read, can_control, can_cpu",
        [
            (("gpu-read",), True, False, False),
            (("gpu-control",), False, True, False),
            (("cpu-read",), False, False, True),
            (("gpu-read", "gpu-control", "cpu-read"), True, True, True),
            ((), False, False, False),
        ],
    )
    def test_auth_with_scopes(self, mock_zeusd, scopes, can_read, can_control, can_cpu):
        """All API groups enabled, but token limits what's accessible."""
        server = mock_zeusd(
            auth_required=True,
            token="valid",
            whoami_scopes=scopes,
        )
        client = ZeusdClient(server.config)
        assert client.can_read_gpu == can_read
        assert client.can_control_gpu == can_control
        assert client.can_read_cpu == can_cpu

    def test_auth_error_disables_all(self, mock_zeusd):
        server = mock_zeusd(
            enabled_api_groups=("gpu-control", "gpu-read", "cpu-read"),
            auth_required=True,
            token=None,
        )
        client = ZeusdClient(server.config)
        assert not client.can_read_gpu
        assert not client.can_control_gpu
        assert not client.can_read_cpu

    def test_scope_without_api_group(self, mock_zeusd):
        """Token grants gpu-read, but the server doesn't enable that group."""
        server = mock_zeusd(
            enabled_api_groups=("gpu-control",),
            auth_required=True,
            token="valid",
            whoami_scopes=("gpu-read", "gpu-control"),
        )
        client = ZeusdClient(server.config)
        assert not client.can_read_gpu
        assert client.can_control_gpu


# ---------------------------------------------------------------------------
# ZeusdClient GPU read
# ---------------------------------------------------------------------------


class TestZeusdClientGpuRead:
    def test_get_gpu_energy(self, mock_zeusd):
        server = mock_zeusd(gpu_ids=(0, 2))
        client = ZeusdClient(server.config)
        result = client.get_gpu_energy([0, 2])
        assert result == {0: 100000, 2: 120000}
        assert server.last_params()["gpu_ids"] == "0,2"

    def test_get_gpu_power_all(self, mock_zeusd):
        server = mock_zeusd(gpu_ids=(0, 1))
        client = ZeusdClient(server.config)
        snap = client.get_gpu_power()
        assert isinstance(snap, GpuPowerSnapshot)
        assert snap.timestamp_ms == 1700000000000
        assert snap.power_mw == {0: 75000, 1: 76000}
        assert "gpu_ids" not in server.last_params()

    def test_get_gpu_power_filtered(self, mock_zeusd):
        server = mock_zeusd(gpu_ids=(0, 1, 2, 3))
        client = ZeusdClient(server.config)
        client.get_gpu_power([1, 3])
        assert server.last_params()["gpu_ids"] == "1,3"

    def test_get_gpu_power_error(self, mock_zeusd):
        server = mock_zeusd(
            endpoint_errors={"/gpu/get_power": 500},
        )
        client = ZeusdClient(server.config)
        with pytest.raises(ZeusdError, match="get_gpu_power"):
            client.get_gpu_power()


# ---------------------------------------------------------------------------
# ZeusdClient GPU control
# ---------------------------------------------------------------------------


class TestZeusdClientGpuControl:
    def test_set_power_limit(self, mock_zeusd):
        server = mock_zeusd()
        client = ZeusdClient(server.config)
        client.set_power_limit([0, 1], 200000, block=True)
        params = server.last_params()
        assert params["gpu_ids"] == "0,1"
        assert params["power_limit_mw"] == "200000"
        assert params["block"] == "true"

    def test_set_power_limit_nonblocking(self, mock_zeusd):
        server = mock_zeusd()
        client = ZeusdClient(server.config)
        client.set_power_limit([0], 150000, block=False)
        assert server.last_params()["block"] == "false"

    def test_set_persistence_mode(self, mock_zeusd):
        server = mock_zeusd()
        client = ZeusdClient(server.config)
        client.set_persistence_mode([0], enabled=True)
        params = server.last_params()
        assert params["gpu_ids"] == "0"
        assert params["enabled"] == "true"
        assert params["block"] == "true"

    def test_set_persistence_mode_disable(self, mock_zeusd):
        server = mock_zeusd()
        client = ZeusdClient(server.config)
        client.set_persistence_mode([0], enabled=False)
        assert server.last_params()["enabled"] == "false"

    def test_set_gpu_locked_clocks(self, mock_zeusd):
        server = mock_zeusd()
        client = ZeusdClient(server.config)
        client.set_gpu_locked_clocks([0, 1], 300, 1200)
        params = server.last_params()
        assert params["gpu_ids"] == "0,1"
        assert params["min_clock_mhz"] == "300"
        assert params["max_clock_mhz"] == "1200"

    def test_reset_gpu_locked_clocks(self, mock_zeusd):
        server = mock_zeusd()
        client = ZeusdClient(server.config)
        client.reset_gpu_locked_clocks([2, 3])
        params = server.last_params()
        assert params["gpu_ids"] == "2,3"
        assert params["block"] == "true"

    def test_set_mem_locked_clocks(self, mock_zeusd):
        server = mock_zeusd()
        client = ZeusdClient(server.config)
        client.set_mem_locked_clocks([0], 400, 1600, block=False)
        params = server.last_params()
        assert params["gpu_ids"] == "0"
        assert params["min_clock_mhz"] == "400"
        assert params["max_clock_mhz"] == "1600"
        assert params["block"] == "false"

    def test_reset_mem_locked_clocks(self, mock_zeusd):
        server = mock_zeusd()
        client = ZeusdClient(server.config)
        client.reset_mem_locked_clocks([0, 1, 2, 3])
        assert server.last_params()["gpu_ids"] == "0,1,2,3"

    def test_control_error(self, mock_zeusd):
        server = mock_zeusd(
            endpoint_errors={"/gpu/set_power_limit": 403},
        )
        client = ZeusdClient(server.config)
        with pytest.raises(ZeusdError, match="set_power_limit"):
            client.set_power_limit([0], 200000)


# ---------------------------------------------------------------------------
# ZeusdClient CPU read
# ---------------------------------------------------------------------------


class TestZeusdClientCpuRead:
    def test_get_cpu_energy(self, mock_zeusd):
        server = mock_zeusd(cpu_ids=(0, 1), dram_available=(True, False))
        client = ZeusdClient(server.config)
        result = client.get_cpu_energy([0, 1])
        assert result == {
            0: CpuEnergyResult(cpu_energy_uj=500000, dram_energy_uj=100000),
            1: CpuEnergyResult(cpu_energy_uj=550000, dram_energy_uj=None),
        }
        params = server.last_params()
        assert params["cpu_ids"] == "0,1"
        assert params["cpu"] == "true"
        assert params["dram"] == "true"

    def test_get_cpu_energy_cpu_only(self, mock_zeusd):
        server = mock_zeusd()
        client = ZeusdClient(server.config)
        client.get_cpu_energy([0], cpu=True, dram=False)
        params = server.last_params()
        assert params["cpu"] == "true"
        assert params["dram"] == "false"

    def test_get_cpu_power_all(self, mock_zeusd):
        server = mock_zeusd(cpu_ids=(0, 1), dram_available=(True, False))
        client = ZeusdClient(server.config)
        snap = client.get_cpu_power()
        assert isinstance(snap, CpuPowerSnapshot)
        assert snap.timestamp_ms == 1700000000000
        assert snap.power_mw == {
            0: CpuDramPower(cpu_mw=50000, dram_mw=10000),
            1: CpuDramPower(cpu_mw=55000, dram_mw=None),
        }
        assert "cpu_ids" not in server.last_params()

    def test_get_cpu_power_filtered(self, mock_zeusd):
        server = mock_zeusd()
        client = ZeusdClient(server.config)
        client.get_cpu_power([0])
        assert server.last_params()["cpu_ids"] == "0"

    def test_get_cpu_power_error(self, mock_zeusd):
        server = mock_zeusd(endpoint_errors={"/cpu/get_power": 500})
        client = ZeusdClient(server.config)
        with pytest.raises(ZeusdError, match="get_cpu_power"):
            client.get_cpu_power()


# ---------------------------------------------------------------------------
# ZeusdClient server utilities
# ---------------------------------------------------------------------------


class TestZeusdClientServer:
    def test_get_time(self, mock_zeusd):
        server = mock_zeusd()
        client = ZeusdClient(server.config)
        assert client.get_time() == 1700000000.0

    def test_get_time_error(self, mock_zeusd):
        server = mock_zeusd(endpoint_errors={"/time": 500})
        client = ZeusdClient(server.config)
        with pytest.raises(ZeusdError, match="get_time"):
            client.get_time()

    def test_endpoint_delegates(self, mock_zeusd):
        server = mock_zeusd()
        client = ZeusdClient(server.config)
        assert client.endpoint == "testhost:4938"

    def test_url_delegates(self, mock_zeusd):
        server = mock_zeusd()
        client = ZeusdClient(server.config)
        assert client.url("/foo") == "http://testhost:4938/foo"


# ---------------------------------------------------------------------------
# require_capabilities
# ---------------------------------------------------------------------------


class TestRequireCapabilities:
    def test_all_met(self, mock_zeusd):
        server = mock_zeusd()
        client = ZeusdClient(server.config)
        require_capabilities(
            client,
            read_gpu=True,
            control_gpu=True,
            read_cpu=True,
            gpu_ids=[0, 1],
            cpu_ids=[0],
        )

    def test_auth_error_raises_auth_error(self, mock_zeusd):
        server = mock_zeusd(auth_required=True, token=None)
        client = ZeusdClient(server.config)
        with pytest.raises(ZeusdAuthError):
            require_capabilities(client, read_gpu=True)

    def test_api_group_not_enabled(self, mock_zeusd):
        server = mock_zeusd(enabled_api_groups=("gpu-control",))
        client = ZeusdClient(server.config)
        with pytest.raises(ZeusdCapabilityError, match="not enabled"):
            require_capabilities(client, read_gpu=True)

    def test_scope_not_granted(self, mock_zeusd):
        server = mock_zeusd(
            auth_required=True,
            token="valid",
            whoami_scopes=("cpu-read",),
        )
        client = ZeusdClient(server.config)
        with pytest.raises(ZeusdCapabilityError, match="lacks required scope"):
            require_capabilities(client, read_gpu=True)

    def test_gpu_ids_missing(self, mock_zeusd):
        server = mock_zeusd(gpu_ids=(0, 1))
        client = ZeusdClient(server.config)
        with pytest.raises(ZeusdCapabilityError, match="GPU indices"):
            require_capabilities(client, gpu_ids=[0, 1, 7])

    def test_cpu_ids_missing(self, mock_zeusd):
        server = mock_zeusd(cpu_ids=(0,))
        client = ZeusdClient(server.config)
        with pytest.raises(ZeusdCapabilityError, match="CPU indices"):
            require_capabilities(client, cpu_ids=[0, 3])

    def test_multiple_errors(self, mock_zeusd):
        server = mock_zeusd(
            enabled_api_groups=(),
            gpu_ids=(0,),
            cpu_ids=(),
            dram_available=(),
        )
        client = ZeusdClient(server.config)
        with pytest.raises(ZeusdCapabilityError) as exc_info:
            require_capabilities(
                client,
                read_gpu=True,
                control_gpu=True,
                read_cpu=True,
                gpu_ids=[0, 5],
                cpu_ids=[0],
            )
        msg = str(exc_info.value)
        assert "gpu-read" in msg
        assert "gpu-control" in msg
        assert "cpu-read" in msg
        assert "GPU indices" in msg
        assert "CPU indices" in msg

    def test_no_requirements_always_passes(self, mock_zeusd):
        server = mock_zeusd(
            enabled_api_groups=(),
            gpu_ids=(),
            cpu_ids=(),
            dram_available=(),
        )
        client = ZeusdClient(server.config)
        require_capabilities(client)

    @pytest.mark.parametrize(
        "groups, scopes, read_gpu, control_gpu, read_cpu",
        [
            # Subset of groups enabled, matching scopes should pass.
            (("gpu-read",), ("gpu-read",), True, False, False),
            (("gpu-control",), ("gpu-control",), False, True, False),
            (("cpu-read",), ("cpu-read",), False, False, True),
            # All groups, all scopes.
            (
                ("gpu-control", "gpu-read", "cpu-read"),
                ("gpu-control", "gpu-read", "cpu-read"),
                True,
                True,
                True,
            ),
        ],
    )
    def test_auth_group_scope_combos_pass(
        self,
        mock_zeusd,
        groups,
        scopes,
        read_gpu,
        control_gpu,
        read_cpu,
    ):
        server = mock_zeusd(
            enabled_api_groups=groups,
            auth_required=True,
            token="valid",
            whoami_scopes=scopes,
        )
        client = ZeusdClient(server.config)
        require_capabilities(
            client,
            read_gpu=read_gpu,
            control_gpu=control_gpu,
            read_cpu=read_cpu,
        )
