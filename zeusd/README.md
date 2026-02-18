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

### UDS mode (GPU management)

For local GPU management, `zeusd` listens on a Unix domain socket:

```sh
sudo zeusd --socket-path /var/run/zeusd.sock --socket-permissions 666
```

To allow the Zeus Python library to recognize that `zeusd` is available, set:

```sh
export ZEUSD_SOCK_PATH=/var/run/zeusd.sock
```

When Zeus detects `ZEUSD_SOCK_PATH`, it'll automatically instantiate the right GPU backend and relay privileged GPU management method calls to `zeusd`.

### TCP mode (power streaming)

For remote power streaming, start `zeusd` in TCP mode:

```sh
sudo zeusd --mode tcp --tcp-bind-address 0.0.0.0:4938
```

Then query power readings via HTTP:

```sh
# One-shot GPU power reading
curl http://localhost:4938/gpu/power

# One-shot CPU power reading (RAPL)
curl http://localhost:4938/cpu/power

# SSE stream of GPU power (Ctrl-C to stop)
curl -N http://localhost:4938/gpu/power/stream

# SSE stream of CPU power
curl -N http://localhost:4938/cpu/power/stream

# Filter to specific devices
curl "http://localhost:4938/gpu/power?gpu_ids=0,1"
curl "http://localhost:4938/cpu/power?cpu_ids=0"
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
