# Zeus Daemon

CPU energy monitoring and GPU/CPU power configuration can require privileges that an ML application should not need.
`zeusd` exposes a scoped HTTP API and holds the device permissions needed for the selected features.
Written in Rust, it adds only microseconds of overhead.

Reach for `zeusd` when you need privilege isolation for GPU configuration, CPU/DRAM energy from unprivileged code, or distributed power monitoring across nodes. For a local privileged process, [`ZeusMonitor`][zeus.monitor.ZeusMonitor] talks to NVML directly.

## Platform support

- **Linux:** UDS default. All API groups work with NVIDIA GPUs through NVML or AMD GPUs through AMD SMI, plus RAPL for CPUs.
- **Windows:** named pipe default. NVML only -- `cpu-read` and `cpu-control` are rejected at startup since RAPL is Linux-only. Python clients must use `--mode tcp` (no `httpx` transport for named pipes yet).

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

## Deployment

See [Deployment](deployment.md) for feature prerequisites, permissions, CPU reset storage, and Native, systemd, and Docker instructions.
Select API groups with `--enable`; Linux enables all four by default, while Windows enables only GPU groups.

| Group | Features |
|---|---|
| `gpu-read` | GPU monitoring and configuration queries |
| `gpu-control` | GPU power limits, locked clocks, and persistence mode |
| `cpu-read` | CPU/DRAM monitoring and CPU power-limit queries (Linux) |
| `cpu-control` | CPU power limits, time windows, and explicit resets (Linux) |

`/discover`, `/time`, and `/auth/whoami` are always available.
Enabling an API group does not grant its device permissions; see [feature requirements](deployment.md#feature-requirements-and-permissions).

## Python integration

Set one of these in the application's environment:

```sh
export ZEUSD_SOCK_PATH=/run/zeusd/zeusd.sock     # UDS (Unix)
export ZEUSD_HOST_PORT=node1:4938                # TCP
```

When set, [`NVIDIAGPUs`][zeus.device.gpu.nvidia.NVIDIAGPUs], [`AMDGPUs`][zeus.device.gpu.amd.AMDGPUs], and [`RAPLCPUs`][zeus.device.cpu.rapl.RAPLCPUs] auto-switch to [`ZeusdNVIDIAGPU`][zeus.device.gpu.nvidia.ZeusdNVIDIAGPU] / [`ZeusdAMDGPU`][zeus.device.gpu.amd.ZeusdAMDGPU] / [`ZeusdRAPLCPU`][zeus.device.cpu.rapl.ZeusdRAPLCPU] backends; privileged GPU calls and CPU/DRAM reads are relayed through the daemon transparently.

For lower-level access: [`ZeusdClient`][zeus.utils.zeusd.ZeusdClient] is a typed wrapper over every [HTTP endpoint](api.md), and [`require_capabilities`][zeus.utils.zeusd.require_capabilities] fails fast if the daemon's capabilities don't match what your code needs.

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
On ROCm older than 7.2, `zeusd` serializes AMD SMI calls to work around a thread-safety bug in `libamd_smi` that was fixed upstream in ROCm 7.2.[^1]

[^1]: AMD SMI read operations are typically very fast and the serialization overhead is negligible, but write operations that take long might experience noticeable slowdowns when done concurrently; in that case, upgrade AMD SMI.

On some AMD GPUs, the driver's cumulative energy counter advances at the wrong rate; see [ROCm/amdsmi #38](https://github.com/ROCm/amdsmi/issues/38).
At startup, `zeusd` integrates power over 0.5 seconds, compares it with the counter delta, and marks GPUs that fail validation as `cumulative_energy_available: false`.

MI250 and MI250X are dual-die GPUs, and from AMD SMI, each die looks like an individual GPU, one even indexed and the other odd indexed.
The driver reports the combined power of both dies on the even-indexed die, and nothing (either unavailable or zero) on the odd-indexed die.

Per AMD SMI's platform support, clock locking and the energy counter work only on bare-metal Linux.
Power capping additionally works on virtualization hosts and single-VF (SR-IOV) guests, but not on multi-VF or Windows guests.
On virtualized AMD GPUs, these operations return 400.

The daemon searches for `libamd_smi.so` once at startup: `/opt/rocm/lib`, then the newest `/opt/rocm-*/lib`, then the dynamic loader paths.
To pin a specific installation, set `ROCM_PATH` (a ROCm installation root, e.g., `/opt/rocm-7.2.0`) or `AMDSMI_LIB_DIR` (a directory directly containing `libamd_smi.so.<abi>`, e.g., `/opt/rocm-7.2.0/lib`); under systemd, in `/etc/default/zeusd`.

## Troubleshooting

- **Python doesn't pick up `zeusd`.** Confirm `ZEUSD_SOCK_PATH` or `ZEUSD_HOST_PORT` is in the *application's* environment (not just the shell that started the daemon). Then run `python -m zeus.show_env`.
- **`Permission denied` on the UDS socket.** Clients need write access. The default `--socket-permissions 666` grants everyone; use `--socket-uid`/`--socket-gid` to scope tighter.
- **No CPU power limits on AMD.** AMD CPUs expose no RAPL power limits. On AMD EPYC CPUs, `sudo modprobe amd_hsmp` creates `/dev/hsmp`, which exposes the socket power limit; restart `zeusd` afterwards. If the module's kernel log says HSMP is disabled, enable it in the BIOS.
- **Intel hardware-range queries or time-window changes fail.** Check [feature requirements](deployment.md#feature-requirements-and-permissions) and the daemon's startup and request logs; energy monitoring, current-limit queries, and power-limit changes do not require MSR access.
- **CPU reset baseline unavailable or mismatched.** Check the [baseline requirements](deployment.md#cpu-reset-baseline) and the error in the daemon logs before changing stored baselines.
- **AMD GPUs not detected.** GPU backends are probed once at startup, so `zeusd` must start after the `amdgpu` driver is loaded (order the systemd unit accordingly, or restart the daemon).
- **AMD SMI startup fails with `AMDSMI_STATUS_UNEXPECTED_DATA` (error 43).** The AMD SMI library is older than the GPU it is reading (e.g., ROCm 6.4 userspace on an MI300X). Point `ROCM_PATH` or `AMDSMI_LIB_DIR` at a ROCm release that supports the GPU.
- **Logs.** `journalctl -u zeusd -f` under systemd; stderr otherwise.
