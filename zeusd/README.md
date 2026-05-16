# Zeus daemon (`zeusd`)

`zeusd` is a system daemon that runs with admin privileges and exposes HTTP API endpoints for GPU management and GPU/CPU power streaming.

## Problem

Energy optimizers in Zeus need to change the GPU's configurations including its power limit or frequency, which requires the Linux security capability `SYS_ADMIN` (effectively `sudo`) or admin elevation on Windows.
It's not a good idea to grant the entire application such strong privileges just to be able to change GPU configurations.

Additionally, monitoring GPU and CPU power across multiple nodes in a cluster requires a lightweight, always-on service that can stream readings to remote clients.

## Solution

`zeusd` runs as a privileged daemon process on the node and provides:

- **GPU management endpoints** that wrap privileged NVML methods, so unprivileged applications can change GPU configuration on their behalf.
- **GPU power streaming** via SSE (Server-Sent Events) using NVML instant power readings.
- **CPU power streaming** via SSE using RAPL energy counters (Intel and modern AMD CPUs, Linux only).

Power polling is demand-driven: `zeusd` only reads from hardware while at least one client is connected, so idle endpoints consume no resources.

To make this as low latency as possible, `zeusd` was written in Rust.

## Platform support

**Linux** is the primary deployment target. Full GPU control and monitoring via NVML, CPU package and DRAM power via Intel RAPL (`/sys/class/powercap`). UDS is the default local transport.

**Windows** runs the GPU control and monitoring path against `nvml.dll`. CPU RAPL is unavailable (the kernel powercap interface is Linux-specific), so `cpu-read` is omitted from the default `--enable` and rejected at startup if explicitly added. Default local transport is a Windows named pipe with a configurable DACL; see [Named Pipe mode](#named-pipe-mode-windows-only).

**macOS** is not supported: the daemon compiles and runs but neither NVML nor RAPL is available there, so it reports zero devices.

## How to use `zeusd`

First, install `zeusd`:

```sh
cargo install zeusd
```

All three transports (UDS, TCP, named pipe) serve the same HTTP API. The only difference is the local transport layer.

### UDS mode (Unix only)

UDS (Unix domain socket) mode is the default on Unix. It's intended for local communication between processes on the same node.

```sh
sudo zeusd serve --socket-path /run/zeusd/zeusd.sock --socket-permissions 666
```

To allow the Zeus Python library to recognize that `zeusd` is available, set:

```sh
export ZEUSD_SOCK_PATH=/run/zeusd/zeusd.sock
```

When Zeus detects `ZEUSD_SOCK_PATH`, it'll automatically instantiate the right GPU backend and relay privileged GPU management method calls to `zeusd`.

### Named Pipe mode (Windows only)

Named Pipe mode is the default on Windows. It plays the same local-IPC role as UDS does on Unix, using a filesystem-style path under `\\.\pipe\`.

```powershell
# In an elevated PowerShell:
zeusd serve --pipe-name \\.\pipe\zeusd
```

The default pipe name is `\\.\pipe\zeusd`. Up to 254 concurrent client connections are accepted (Windows' `PIPE_UNLIMITED_INSTANCES` = 255 is rejected by Tokio's builder, so 254 is the practical cap).

`actix-web`'s built-in `HttpServer` does not support Windows named pipes, so on the named-pipe path zeusd drives `actix-http`'s `H1Service` over each connected `NamedPipeServer` on a single-threaded `LocalSet`. `--num-workers` is therefore ignored in this mode.

Access control is enforced via the pipe's DACL, configurable through `--pipe-sddl`. The default `D:(A;;GRGW;;;AU)` grants read+write to all authenticated users, analogous to `--socket-permissions 666` for UDS; pass any valid SDDL string to tighten or widen the audience.

**Python client integration.** The Zeus Python library (`zeus.utils.zeusd.ZeusdConfig` and the `ZeusdNVIDIAGPU` / `PowerStreamingClient` it backs) currently constructs HTTP transports for TCP and Unix domain sockets only; there is no `httpx` transport for Windows named pipes yet. To use `zeusd` from a Zeus Python application on Windows, run the daemon in TCP mode (`--mode tcp`) and point clients at `ZEUSD_HOST_PORT`. Named-pipe mode is fully usable from non-Python clients (curl in WSL, .NET `NamedPipeClientStream`, custom Rust/Go clients, etc.).

### TCP mode

TCP mode exposes the same API over a TCP socket, making it accessible from remote hosts. This is useful for cluster-wide power monitoring and works on all platforms.

```sh
sudo zeusd serve --mode tcp --tcp-bind-address 0.0.0.0:4938
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
from zeus.utils.zeusd import ZeusdConfig
from zeus.monitor.power_streaming import PowerStreamingClient

client = PowerStreamingClient(
    servers=[
        ZeusdConfig.tcp(host="node1", port=4938, gpu_indices=[0, 1, 2, 3]),
        ZeusdConfig.uds(socket_path="/run/zeusd/zeusd.sock", gpu_indices=[0, 1, 2, 3]),
    ],
)

# Get current power reading once
readings = client.get_power()

# Continuously stream power readings
for power_reading in client:
    print(power_reading)
```

See the [Distributed Power Measurement and Aggregation](https://ml.energy/zeus/measure/#distributed-power-measurement-and-aggregation) section in our documentation for more details.

### API groups

`zeusd` organizes its endpoints into API groups that can be selectively enabled with the `--enable` flag. By default Linux enables all three groups (`gpu-control,gpu-read,cpu-read`); Windows omits `cpu-read`.

| Group | Endpoints | Requires root |
|-------|-----------|:---:|
| `gpu-control` | `POST /gpu/set_persistence_mode` | Yes |
| | `POST /gpu/set_power_limit` | |
| | `POST /gpu/set_gpu_locked_clocks` | |
| | `POST /gpu/reset_gpu_locked_clocks` | |
| | `POST /gpu/set_mem_locked_clocks` | |
| | `POST /gpu/reset_mem_locked_clocks` | |
| `gpu-read` | `GET /gpu/get_power` | No |
| | `GET /gpu/stream_power` | |
| | `GET /gpu/get_cumulative_energy` | |
| `cpu-read` (Linux only) | `GET /cpu/get_cumulative_energy` | Yes |
| | `GET /cpu/get_power` | |
| | `GET /cpu/stream_power` | |

The following endpoints are always available regardless of which groups are enabled:
- `GET /discover`
- `GET /time`

If a group that requires root is enabled but the daemon is not running as root, it will exit immediately with an error. On Windows there is no fail-fast admin check; NVML write calls fall back to NVML's own `NoPermission` if the daemon lacks admin, which surfaces as HTTP 403 to clients.

### Authentication

`zeusd` supports optional per-user JWT authentication. When `--signing-key-path` is provided, all endpoints except `/discover` and `/time` require a valid `Authorization: Bearer <token>` header.

**Setting up a signing key:**

```sh
# Generate a 32-byte signing key (shared across all daemons in a cluster)
openssl rand -base64 32 > /etc/zeusd/signing.key
chmod 600 /etc/zeusd/signing.key
```

**Starting the daemon with auth:**

```sh
sudo zeusd serve --mode tcp --tcp-bind-address 0.0.0.0:4938 --signing-key-path /etc/zeusd/signing.key
```

**Issuing tokens:**

```sh
# Token with 7-day expiry and GPU read scope
zeusd token issue \
    --signing-key-path /etc/zeusd/signing.key \
    --user alice \
    --scope gpu-read \
    --expires 7d

# Token with multiple scopes and no expiry
zeusd token issue \
    --signing-key-path /etc/zeusd/signing.key \
    --user alice \
    --scope gpu-read,gpu-control,cpu-read \
    --expires never
```

`--expires` accepts human-readable durations (`1h`, `7d`, `30d`) or `never`/`0` for tokens that never expire.

**Using tokens with Python clients:**

Set the `ZEUSD_TOKEN` environment variable, or pass the token directly:

```sh
export ZEUSD_TOKEN="eyJ..."
```

Python clients (`ZeusdNVIDIAGPU`, `ZeusdRAPLCPU`, `PowerStreamingClient`) automatically check `/discover` to determine whether auth is required. If auth is required and no token is available, an error is raised with a clear message.

**Using tokens with curl:**

```sh
curl -H "Authorization: Bearer $ZEUSD_TOKEN" http://localhost:4938/gpu/get_power
```

When no `--signing-key-path` is provided, the daemon runs without authentication and all endpoints are freely accessible. The `/discover` endpoint always reports `auth_required: true` or `false` so clients can adapt.

## Testing

Unit and integration tests run via `cargo test` inside `zeusd/`. CI runs the matrix on every push.

For Windows-specific end-to-end coverage with a real NVIDIA GPU, two helper scripts live in `zeusd/scripts/`:

- `test-windows-gpu.sh` provisions a `g4dn.xlarge` on AWS, builds `zeusd` from a Git ref, runs TCP + named-pipe smoke tests, a PyTorch matmul load with NVML power-limit / locked-clocks round-trips, and an SDDL ACL test that drives privileged NVML writes from an unprivileged client through the elevated daemon. Tears its own resources down via a trap. Pass `-h` for options.
- `cleanup-test-aws.sh` lists or deletes resources tagged `zeusd-test` (used to recover from crashed traps or stale orphans on a shared AWS account). Default mode is list-only; deletion requires an explicit scope (`--tag-value`, `--older-than`, or `--all`). Pass `-h` for the full usage.

Both scripts require `aws` CLI v2 with valid credentials and `jq`. Resources are tagged with a unique per-run value (`zeusd-test-<utc-second>-<pid>-<6 hex chars>`), so concurrent runs by multiple devs on the same AWS account don't collide and a scoped cleanup affects only its own run.

## API Reference

### Discovery

#### `GET /discover`

Returns available devices, capabilities, and enabled API groups.

Response:

| Field | Type | Description |
|-------|------|-------------|
| `gpu_ids` | `int[]` | Available GPU indices |
| `cpu_ids` | `int[]` | Available CPU indices |
| `dram_available` | `bool[]` | Per-CPU DRAM energy support (indexed by position in `cpu_ids`) |
| `enabled_api_groups` | `string[]` | API groups enabled on this instance |
| `auth_required` | `bool` | Whether JWT authentication is required |

Example response:
```json
{
  "gpu_ids": [0, 1, 2, 3],
  "cpu_ids": [0, 1],
  "dram_available": [true, false],
  "enabled_api_groups": ["gpu-control", "gpu-read", "cpu-read"],
  "auth_required": false
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
The Zeus daemon manages and monitors compute devices on the node

Usage: zeusd <COMMAND>

Commands:
  serve  Start the Zeus daemon
  token  Token management
  help   Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

Note: a few flags are platform-conditional. The `--mode` default and the UDS / named-pipe options below only appear on the platforms where they apply.

```console
$ zeusd serve --help     # Linux variant shown; Windows omits --socket-* and adds --pipe-name
Start the Zeus daemon

Usage: zeusd serve [OPTIONS]

Options:
      --mode <MODE>
          Operating mode (default depends on platform: `uds` on Unix, `named-pipe` on Windows)

          Possible values:
          - uds:         Unix domain socket (Unix only)
          - tcp:         TCP
          - named-pipe:  Windows named pipe (Windows only)

      --socket-path <SOCKET_PATH>          # Unix only
          [UDS mode] Path to the socket Zeusd will listen on

          [default: /run/zeusd/zeusd.sock]

      --socket-permissions <SOCKET_PERMISSIONS>          # Unix only
          [UDS mode] Permissions for the socket file to be created

          [default: 666]

      --socket-uid <SOCKET_UID>            # Unix only
          [UDS mode] UID to chown the socket file to

      --socket-gid <SOCKET_GID>            # Unix only
          [UDS mode] GID to chown the socket file to

      --pipe-name <PIPE_NAME>              # Windows only
          [Named pipe mode] Named pipe Zeusd will listen on

          [default: \\.\pipe\zeusd]

      --tcp-bind-address <TCP_BIND_ADDRESS>
          [TCP mode] Address to bind to

          [default: 127.0.0.1:4938]

      --num-workers <NUM_WORKERS>
          Number of worker threads to use. Default is the number of logical CPUs.
          Ignored in named-pipe mode (which uses a single-threaded LocalSet)

      --gpu-power-poll-hz <GPU_POWER_POLL_HZ>
          GPU power polling frequency in Hz for the streaming endpoint

          [default: 20]

      --cpu-power-poll-hz <CPU_POWER_POLL_HZ>
          CPU RAPL power polling frequency in Hz for the streaming endpoint

          [default: 10]

      --enable <ENABLE>
          API groups to enable. Each group exposes a set of HTTP endpoints. Groups that require root will cause the daemon to exit at startup if it is not running as root. Linux defaults to all three; Windows omits `cpu-read`

          [default: gpu-control gpu-read cpu-read]   # Linux
          [default: gpu-control gpu-read]            # Windows

          Possible values:
          - gpu-control: GPU control operations (set power limit, clocks, persistence mode). Requires root
          - gpu-read:    GPU read operations (power reading, energy consumption)
          - cpu-read:    CPU RAPL read operations (energy, power). Requires root

      --signing-key-path <SIGNING_KEY_PATH>
          Path to the HMAC-SHA256 signing key file for JWT authentication. If not provided, authentication is disabled

  -h, --help
          Print help (see a summary with '-h')
```

```console
$ zeusd token issue --help
Issue a new JWT token for a user

Usage: zeusd token issue [OPTIONS] --signing-key-path <SIGNING_KEY_PATH> --user <USER> --expires <EXPIRES>

Options:
      --signing-key-path <SIGNING_KEY_PATH>
          Path to the HMAC-SHA256 signing key file

      --user <USER>
          User identity to embed in the token (the `sub` claim)

      --scope <SCOPE>
          API group scopes to grant. Comma-separated

          Possible values:
          - gpu-control: GPU control operations (set power limit, clocks, persistence mode). Requires root
          - gpu-read:    GPU read operations (power reading, energy consumption)
          - cpu-read:    CPU RAPL read operations (energy, power). Requires root

      --expires <EXPIRES>
          Token lifetime. Human-readable duration (e.g., "1h", "7d", "30d"). Use "never" for tokens that do not expire

  -h, --help
          Print help (see a summary with '-h')
```
