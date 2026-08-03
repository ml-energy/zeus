# Zeus Daemon

Intel RAPL and GPU power configuration (power limit, locked clocks, persistence mode) are privileged operations: root on Linux, admin elevation on Windows. Granting an entire ML application that level of system privilege is too much. So `zeusd` runs in privileged mode and exposes a limited, scoped HTTP API to unprivileged applications. Written in Rust, so going through the daemon adds only microseconds of overhead.

Reach for `zeusd` when you need privilege isolation for GPU configuration, CPU/DRAM energy from unprivileged code, or distributed power monitoring across nodes. For a local privileged process, [`ZeusMonitor`][zeus.monitor.ZeusMonitor] talks to NVML directly.

## Platform support

- **Linux:** UDS default. All API groups work with NVIDIA GPUs through NVML or AMD GPUs through AMD SMI, plus RAPL for CPUs.
- **Windows:** named pipe default. NVML only -- `cpu-read` is rejected at startup since RAPL is Linux-only. Python clients must use `--mode tcp` (no `httpx` transport for named pipes yet).

NVIDIA GPU support loads NVML at runtime.
AMD GPU support loads AMD SMI at runtime, but the ABI is not stable across versions.
At the moment, ABI 24, 25, and 26 are supported (ROCm 6.3 to latest), and we will add support for future versions as they stabilize and release.

## Install

```sh
cargo install zeusd
```

NVML and AMD SMI support are both enabled by default.
To build with only one backend:

```sh
cargo install zeusd --no-default-features --features nvml
cargo install zeusd --no-default-features --features amdsmi
```

Linux deployments: see [`zeusd/packaging/systemd/`](https://github.com/ml-energy/zeus/tree/master/zeusd/packaging/systemd){.external} for a hardened systemd unit.

## Running it

```sh
# Linux (UDS, default)
sudo zeusd serve --socket-path /run/zeusd/zeusd.sock --socket-permissions 666

# Windows (named pipe, default; from elevated PowerShell)
zeusd serve --pipe-name \\.\pipe\zeusd

# TCP for cluster-wide monitoring, or for Python clients on Windows
sudo zeusd serve --mode tcp --tcp-bind-address 0.0.0.0:4938
```

Defaults to all API groups on Linux, GPU only on Windows.

The daemon searches for `libamd_smi.so` once at startup: `/opt/rocm/lib`, then the newest `/opt/rocm-*/lib`, then the dynamic loader paths. Setting `AMDSMI_LIB_DIR` (a library directory) or `ROCM_PATH` (a ROCm installation root) restricts the search to that installation, e.g. `ROCM_PATH=/opt/rocm-7.2.0 zeusd serve`.

## API groups

Selectively enable with `--enable`:

| Group | What | Needs root |
|---|---|:---:|
| `gpu-control` | `POST /gpu/{set,reset}_*` (power limit, locked clocks, persistence) | Yes |
| `gpu-read` | `GET /gpu/{get,stream}_power`, `get_cumulative_energy` | No |
| `cpu-read` (Linux) | `GET /cpu/{get,stream}_power`, `get_cumulative_energy` | Yes |

`/discover`, `/time`, and `/auth/whoami` are always available. On Linux, the daemon refuses to start if a root-required group is enabled without root; on Windows there's no admin check, and unprivileged NVML writes surface as HTTP 403.

For read-only monitoring without root: `--enable gpu-read`.

## Python integration

Set one of these in the application's environment:

```sh
export ZEUSD_SOCK_PATH=/run/zeusd/zeusd.sock     # UDS (Unix)
export ZEUSD_HOST_PORT=node1:4938                # TCP
```

When set, [`NVIDIAGPUs`][zeus.device.gpu.nvidia.NVIDIAGPUs], [`AMDGPUs`][zeus.device.gpu.amd.AMDGPUs], and [`RAPLCPUs`][zeus.device.cpu.rapl.RAPLCPUs] auto-switch to [`ZeusdNVIDIAGPU`][zeus.device.gpu.nvidia.ZeusdNVIDIAGPU] / [`ZeusdAMDGPU`][zeus.device.gpu.amd.ZeusdAMDGPU] / [`ZeusdRAPLCPU`][zeus.device.cpu.rapl.ZeusdRAPLCPU] backends; privileged GPU calls and CPU/DRAM reads are relayed through the daemon transparently.

For lower-level access: [`ZeusdClient`][zeus.utils.zeusd.ZeusdClient] is a typed wrapper over every HTTP endpoint, and [`require_capabilities`][zeus.utils.zeusd.require_capabilities] fails fast if the daemon's capabilities don't match what your code needs.

For distributed power streaming across nodes, see [Distributed Power Measurement and Aggregation](../measure/index.md#distributed-power-measurement-and-aggregation).

## Authentication (optional)

JWT with per-user scopes. Skip if running on UDS or a trusted local network.

```sh
# Generate a signing key (shared across daemons in a cluster).
sudo install -d -m 0755 /etc/zeusd
openssl rand -base64 32 | sudo tee /etc/zeusd/signing.key > /dev/null
sudo chmod 600 /etc/zeusd/signing.key

# Start the daemon with auth.
sudo zeusd serve --mode tcp --tcp-bind-address 0.0.0.0:4938 \
    --signing-key-path /etc/zeusd/signing.key

# Issue a 7-day token scoped to GPU read.
zeusd token issue --signing-key-path /etc/zeusd/signing.key \
    --user alice --scope gpu-read --expires 7d
```

`--expires` accepts `1h`, `7d`, `30d`, or `never`. Hand the token to applications via `ZEUSD_TOKEN`, or `-H "Authorization: Bearer ..."` for curl. `/discover` and `/time` never require auth.

## Notes on Platforms

### Windows

NVML's persistence-mode API is Linux-only. On Windows the kernel driver is always loaded, so `POST /gpu/set_persistence_mode?enabled=true` is a 200 no-op (logged once); `enabled=false` returns 400. All other GPU operations behave identically across platforms.

### NVIDIA GPU

The cumulative energy counter requires Volta or newer.
On older GPUs, `cumulative_energy_available` is false in `/discover` and `GET /gpu/get_cumulative_energy` returns 400.

### AMD GPU

GPU indices follow PCI bus order sorted by PCI address, which is the same order used by the `amd-smi` CLI.
The reported GPU name is AMD SMI's market name and can differ across ROCm versions for the same GPU.

On some AMD GPUs, the driver's cumulative energy counter advances at the wrong rate; see [ROCm/amdsmi #38](https://github.com/ROCm/amdsmi/issues/38).
At startup, `zeusd` integrates power over 0.5 seconds, compares it with the counter delta, and marks GPUs that fail validation as `cumulative_energy_available: false`.

MI250 and MI250X are dual-die GPUs, and from AMD SMI, each die looks like an individual GPU, one even indexed and the other odd indexed.
The driver reports the combined power of both dies on the even-indexed die, and nothing (either unavailable or zero) on the odd-indexed die.

Per AMD SMI's platform support, clock locking and the energy counter work only on bare-metal Linux.
Power capping additionally works on virtualization hosts and single-VF (SR-IOV) guests, but not on multi-VF or Windows guests.
On virtualized AMD GPUs, these operations return 400.

When ROCm is outside the default search locations, point the daemon at it in `/etc/default/zeusd`.
`ROCM_PATH` is a ROCm installation root whose `lib/` contains `libamd_smi.so.<abi>`, e.g., `ROCM_PATH=/opt/rocm-7.2.0`.
`AMDSMI_LIB_DIR` is any directory that directly contains `libamd_smi.so.<abi>`, e.g., `AMDSMI_LIB_DIR=/opt/rocm-7.2.0/lib`.

## Troubleshooting

- **Python doesn't pick up `zeusd`.** Confirm `ZEUSD_SOCK_PATH` or `ZEUSD_HOST_PORT` is in the *application's* environment (not just the shell that started the daemon). Then run `python -m zeus.show_env`.
- **`Permission denied` on the UDS socket.** Clients need write access. The default `--socket-permissions 666` grants everyone; use `--socket-uid`/`--socket-gid` to scope tighter.
- **Daemon exits immediately at startup.** On Linux, a root-required group is enabled but `zeusd` isn't running as root. Either `sudo` or `--enable gpu-read`.
- **AMD GPUs not detected.** GPU backends are probed once at startup, so `zeusd` must start after the `amdgpu` driver is loaded (order the systemd unit accordingly, or restart the daemon).
- **Logs.** `journalctl -u zeusd -f` under systemd; stderr otherwise.

## HTTP API reference

The API is the same regardless of transport. Paths shown below are server-relative; prefix with `http://<host>:<port>` over TCP, the UDS socket over UDS, or the named pipe on Windows.

Status codes: `200` success; `400` bad input or unsupported op (e.g., persistence-mode off on Windows); `401` missing/invalid token; `403` insufficient token scope or NVML `NoPermission`; `404` disabled API group or `/auth/*` on a no-auth daemon. Per-device write calls aggregate per-device errors into `{"errors": {"<device_id>": "<message>"}}` with the worst per-device status.

### `GET /discover`

Available devices, capabilities, and enabled API groups. Always available; never requires auth.

```json
{
  "gpus": [
    {"id": 0, "name": "NVIDIA A40", "pci_address": "0000:01:00.0", "cumulative_energy_available": true},
    {"id": 1, "name": "NVIDIA A40", "pci_address": "0000:41:00.0", "cumulative_energy_available": true}
  ],
  "cpus": [
    {"id": 0, "dram_available": true},
    {"id": 1, "dram_available": false}
  ],
  "enabled_api_groups": ["gpu-control", "gpu-read", "cpu-read"],
  "auth_required": false
}
```

`pci_address` is the PCI domain:bus:device.function address, formatted as in `lspci -D`.
`cumulative_energy_available` states whether the GPU has a trustworthy cumulative energy counter; when false, `GET /gpu/get_cumulative_energy` returns 400 for that GPU (see [Notes on Platforms](#notes-on-platforms)).

### `GET /time`

Daemon-side Unix timestamp in milliseconds. Always available.

```json
{"timestamp_ms": 1762000000000}
```

### `GET /auth/whoami`

Authenticated user's identity and scopes. Requires a bearer token. Returns 404 when auth is disabled.

```json
{"sub": "alice", "scopes": ["gpu-read", "gpu-control"], "exp": 1762864200}
```

`exp` is omitted for tokens issued with `--expires never`.

### GPU

All endpoints are under `/gpu`. `gpu_ids` is a comma-separated list of GPU indices: required on writes; optional on reads (omit to apply to / read all GPUs).

Writes (`POST`) also take `block` (bool): `true` waits for completion and reports per-GPU execution errors; `false` dispatches non-blocking and only reports MPSC send errors.

| Method | Path | Extra params / notes |
|---|---|---|
| `POST` | `/gpu/set_power_limit` | `power_limit_mw` |
| `POST` | `/gpu/set_persistence_mode` | `enabled`; AMD GPUs return 400 because persistence mode is an NVML concept (see [Windows notes](#windows)). |
| `POST` | `/gpu/set_gpu_locked_clocks` | `min_clock_mhz`, `max_clock_mhz` |
| `POST` | `/gpu/reset_gpu_locked_clocks` | On AMD GPUs, returns 400 (no per-domain reset exists); use `reset_locked_clocks`. |
| `POST` | `/gpu/set_mem_locked_clocks` | `min_clock_mhz`, `max_clock_mhz` |
| `POST` | `/gpu/reset_mem_locked_clocks` | On AMD GPUs, returns 400 (no per-domain reset exists); use `reset_locked_clocks`. |
| `POST` | `/gpu/reset_locked_clocks` | resets all clock domains |
| `GET`  | `/gpu/get_cumulative_energy` | GPUs whose `cumulative_energy_available` is false in `/discover` return 400. |
| `GET`  | `/gpu/get_power` | one-shot snapshot |
| `GET`  | `/gpu/stream_power` | SSE stream |
| `GET`  | `/gpu/get_power_limit` | -- |
| `GET`  | `/gpu/get_power_limit_constraints` | -- |
| `GET`  | `/gpu/get_persistence_mode` | AMD GPUs return 400 because persistence mode is an NVML concept; always `true` on Windows. |

`get_cumulative_energy` response (keyed by GPU index as string):

```json
{"0": {"energy_mj": 123456}, "1": {"energy_mj": 789012}}
```

`get_power` returns a snapshot keyed by GPU index:

```json
{"timestamp_ms": 1762000000000, "power_mw": {"0": 75000, "1": 120000}}
```

`stream_power` emits one SSE event per GPU sample:

```text
data: {"timestamp_ms": 1762000000000, "gpu_id": 0, "power_mw": 75000}
```

If `gpu_ids` is provided, only those GPUs are polled.

`get_power_limit`, `get_power_limit_constraints`, and `get_persistence_mode` responses (keyed by GPU index as string):

```json
{"0": {"power_limit_mw": 200000}, "1": {"power_limit_mw": 250000}}
{"0": {"min_power_limit_mw": 100000, "max_power_limit_mw": 300000}}
{"0": {"enabled": true}, "1": {"enabled": false}}
```

### CPU

All endpoints are under `/cpu` (Linux only). `cpu_ids` is a comma-separated list of RAPL package indices (the `N` in `/sys/class/powercap/intel-rapl/intel-rapl:N/`, not core or hyperthread IDs); optional on all endpoints (omit to read all CPUs).

| Method | Path | Extra params / notes |
|---|---|---|
| `GET` | `/cpu/get_cumulative_energy` | `cpu` (bool) and `dram` (bool), both required |
| `GET` | `/cpu/get_power` | one-shot snapshot |
| `GET` | `/cpu/stream_power` | SSE stream |

`get_cumulative_energy` response (fields nullable):

```json
{
  "0": {"cpu_energy_uj": 123456, "dram_energy_uj": 78901},
  "1": {"cpu_energy_uj": 234567, "dram_energy_uj": null}
}
```

`get_power` returns a snapshot keyed by CPU index:

```json
{
  "timestamp_ms": 1762000000000,
  "power_mw": {
    "0": {"cpu_mw": 85000, "dram_mw": 12000},
    "1": {"cpu_mw": 78000, "dram_mw": null}
  }
}
```

`stream_power` emits one SSE event per CPU package sample:

```text
data: {"timestamp_ms": 1762000000000, "cpu_id": 0, "cpu_mw": 85000, "dram_mw": 12000}
```

If `cpu_ids` is provided, only those CPU packages are polled.
