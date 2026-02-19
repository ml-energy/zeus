# Zeus daemon (`zeusd`)

`zeusd` is a system daemon that runs with admin privileges and exposes HTTP API endpoints for GPU management and GPU/CPU power streaming.

## Problem

Energy optimizers in Zeus need to change the GPU's configurations including its power limit or frequency, which requires the Linux security capability `SYS_ADMIN` (which is pretty much `sudo`).
However, it's not a good idea to grant the entire application such strong privileges just to be able to change GPU configurations.

Additionally, monitoring GPU and CPU power across multiple nodes in a cluster requires a lightweight, always-on service that can stream readings to remote clients.

## Solution

`zeusd` runs as a privileged daemon process on the node and provides:

- **GPU management endpoints** that wrap privileged NVML methods, so unprivileged applications can change GPU configuration on their behalf.
- **GPU power streaming** via SSE (Server-Sent Events) using NVML instant power readings.
- **CPU power streaming** via SSE using RAPL energy counters (Intel and modern AMD CPUs).

Power polling is demand-driven: `zeusd` only reads from hardware while at least one client is connected, so idle endpoints consume no resources.

To make this as low latency as possible, `zeusd` was written in Rust.

## How to use `zeusd`

First, install `zeusd`:

```sh
cargo install zeusd
```

Both modes (UDS and TCP) serve the same HTTP API. The only difference is the transport layer.

### UDS mode

UDS (Unix domain socket) mode is the default. It's intended for local communication between processes on the same node.

```sh
sudo zeusd --socket-path /var/run/zeusd.sock --socket-permissions 666
```

To allow the Zeus Python library to recognize that `zeusd` is available, set:

```sh
export ZEUSD_SOCK_PATH=/var/run/zeusd.sock
```

When Zeus detects `ZEUSD_SOCK_PATH`, it'll automatically instantiate the right GPU backend and relay privileged GPU management method calls to `zeusd`.

### TCP mode

TCP mode exposes the same API over a TCP socket, making it accessible from remote hosts. This is useful for cluster-wide power monitoring.

```sh
sudo zeusd --mode tcp --tcp-bind-address 0.0.0.0:4938
```

Example queries via curl:

```sh
# Discovery
curl http://localhost:4938/discover

# One-shot GPU power reading
curl http://localhost:4938/gpu/get_power

# One-shot CPU power reading (RAPL)
curl http://localhost:4938/cpu/get_power

# SSE stream of GPU power (Ctrl-C to stop)
curl -N http://localhost:4938/gpu/stream_power

# SSE stream of CPU power
curl -N http://localhost:4938/cpu/stream_power

# Filter to specific devices
curl "http://localhost:4938/gpu/get_power?gpu_ids=0,1"
curl "http://localhost:4938/cpu/get_power?cpu_ids=0"

# GPU management via query params
curl -X POST 'http://localhost:4938/gpu/set_power_limit?gpu_ids=0,1&power_limit_mw=200000&block=true'
```

On the Python side, use `PowerStreamingClient` to connect to one or more `zeusd` instances:

```python
from zeus.monitor.power_streaming import PowerStreamingClient, ZeusdTcpConfig

client = PowerStreamingClient(
    servers=[
        ZeusdTcpConfig(
            host="node1", port=4938,
            gpu_indices=[0, 1, 2, 3],
            cpu_indices=[0],
        ),
        ZeusdTcpConfig(host="node2", port=4938),
    ],
)
readings = client.get_power()
client.stop()
```

See the [Distributed Power Measurement and Aggregation](https://ml.energy/zeus/measure/#distributed-power-measurement-and-aggregation) section in our documentation for more details.

## API Reference

### Discovery

#### `GET /discover`

Returns available devices and capabilities.

Response:

| Field | Type | Description |
|-------|------|-------------|
| `gpu_ids` | `int[]` | Available GPU indices |
| `cpu_ids` | `int[]` | Available CPU indices |
| `dram_available` | `bool[]` | Per-CPU DRAM energy support (indexed by position in `cpu_ids`) |

Example response:
```json
{
  "gpu_ids": [0, 1, 2, 3],
  "cpu_ids": [0, 1],
  "dram_available": [true, false]
}
```

### GPU

All GPU endpoints are under the `/gpu` scope.

#### `POST /gpu/set_power_limit`

Set GPU power management limit.

| Query param | Type | Required | Description |
|-------------|------|----------|-------------|
| `gpu_ids` | `string` | yes | Comma-separated GPU indices |
| `power_limit_mw` | `int` | yes | Power limit in milliwatts |
| `block` | `bool` | yes | Wait for completion |

#### `POST /gpu/set_persistence_mode`

Set GPU persistence mode.

| Query param | Type | Required | Description |
|-------------|------|----------|-------------|
| `gpu_ids` | `string` | yes | Comma-separated GPU indices |
| `enabled` | `bool` | yes | Enable or disable persistence mode |
| `block` | `bool` | yes | Wait for completion |

#### `POST /gpu/set_gpu_locked_clocks`

Lock GPU core clocks to a range.

| Query param | Type | Required | Description |
|-------------|------|----------|-------------|
| `gpu_ids` | `string` | yes | Comma-separated GPU indices |
| `min_clock_mhz` | `int` | yes | Minimum clock in MHz |
| `max_clock_mhz` | `int` | yes | Maximum clock in MHz |
| `block` | `bool` | yes | Wait for completion |

#### `POST /gpu/reset_gpu_locked_clocks`

Reset GPU core locked clocks.

| Query param | Type | Required | Description |
|-------------|------|----------|-------------|
| `gpu_ids` | `string` | yes | Comma-separated GPU indices |
| `block` | `bool` | yes | Wait for completion |

#### `POST /gpu/set_mem_locked_clocks`

Lock GPU memory clocks to a range.

| Query param | Type | Required | Description |
|-------------|------|----------|-------------|
| `gpu_ids` | `string` | yes | Comma-separated GPU indices |
| `min_clock_mhz` | `int` | yes | Minimum clock in MHz |
| `max_clock_mhz` | `int` | yes | Maximum clock in MHz |
| `block` | `bool` | yes | Wait for completion |

#### `POST /gpu/reset_mem_locked_clocks`

Reset GPU memory locked clocks.

| Query param | Type | Required | Description |
|-------------|------|----------|-------------|
| `gpu_ids` | `string` | yes | Comma-separated GPU indices |
| `block` | `bool` | yes | Wait for completion |

#### `GET /gpu/get_cumulative_energy[?gpu_ids=0,1]`

Total energy consumption since driver load (NVML). `gpu_ids` is optional (omit = all GPUs).

Response (map keyed by GPU ID):
```json
{"0": {"energy_mj": 123456}, "1": {"energy_mj": 789012}}
```

#### `GET /gpu/get_power[?gpu_ids=0,1]`

One-shot GPU power reading. `gpu_ids` is optional (omit = all GPUs).

#### `GET /gpu/stream_power[?gpu_ids=0,1]`

SSE stream of GPU power readings. `gpu_ids` is optional (omit = all GPUs).

### CPU

All CPU endpoints are under the `/cpu` scope.

#### `GET /cpu/get_cumulative_energy`

Get cumulative RAPL energy counters.

| Query param | Type | Required | Description |
|-------------|------|----------|-------------|
| `cpu_ids` | `string` | yes | Comma-separated CPU indices |
| `cpu` | `bool` | yes | Include CPU package energy |
| `dram` | `bool` | yes | Include DRAM energy |

Response is a map keyed by CPU ID:
```json
{
  "0": {"cpu_energy_uj": 123456, "dram_energy_uj": 78901},
  "1": {"cpu_energy_uj": 234567, "dram_energy_uj": null}
}
```

#### `GET /cpu/get_power[?cpu_ids=0,1]`

One-shot CPU power reading (computed from RAPL energy deltas). `cpu_ids` is optional (omit = all CPUs).

#### `GET /cpu/stream_power[?cpu_ids=0,1]`

SSE stream of CPU power readings. `cpu_ids` is optional (omit = all CPUs).

### Full help message

```console
$ zeusd --help
The Zeus daemon runs with elevated provileges and communicates with unprivileged Zeus clients to allow them to interact with and control compute devices on the node

Usage: zeusd [OPTIONS]

Options:
      --mode <MODE>
          Operating mode: UDS or TCP

          [default: uds]

          Possible values:
          - uds: Unix domain socket
          - tcp: TCP

      --socket-path <SOCKET_PATH>
          [UDS mode] Path to the socket Zeusd will listen on

          [default: /var/run/zeusd.sock]

      --socket-permissions <SOCKET_PERMISSIONS>
          [UDS mode] Permissions for the socket file to be created

          [default: 666]

      --socket-uid <SOCKET_UID>
          [UDS mode] UID to chown the socket file to

      --socket-gid <SOCKET_GID>
          [UDS mode] GID to chown the socket file to

      --tcp-bind-address <TCP_BIND_ADDRESS>
          [TCP mode] Address to bind to

          [default: 127.0.0.1:4938]

      --allow-unprivileged
          If set, Zeusd will not complain about running as non-root

      --num-workers <NUM_WORKERS>
          Number of worker threads to use. Default is the number of logical CPUs

      --gpu-power-poll-hz <GPU_POWER_POLL_HZ>
          GPU power polling frequency in Hz for the streaming endpoint

          [default: 20]

      --cpu-power-poll-hz <CPU_POWER_POLL_HZ>
          CPU RAPL power polling frequency in Hz for the streaming endpoint

          [default: 10]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```
