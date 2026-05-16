# Zeus daemon (`zeusd`)

`zeusd` is a system daemon that runs with admin privileges and exposes an HTTP API for GPU management and GPU/CPU power streaming. It lets unprivileged applications safely change GPU configuration without granting them `SYS_ADMIN`, and serves cluster-wide power readings over Server-Sent Events.

**Full documentation: <https://ml.energy/zeus/zeusd/>**

## Why

Energy optimizers in Zeus need to change GPU configuration (power limit, clocks, persistence mode), which requires `SYS_ADMIN` on Linux or admin elevation on Windows. Granting an entire ML application those privileges is too much. `zeusd` runs as a single privileged process per node and exposes a minimal set of HTTP endpoints; unprivileged applications relay privileged calls through it.

`zeusd` also serves GPU and CPU power readings via SSE -- demand-driven, so hardware is only polled while a client is connected -- enabling cluster-wide power monitoring without giving every node-side process root.

## Platform support

- **Linux:** UDS default. All API groups work (NVML + RAPL).
- **Windows:** named pipe default. NVML only; `cpu-read` is rejected at startup since RAPL is Linux-only. Python clients must use `--mode tcp`.

## Install

```sh
cargo install zeusd
```

For a hardened systemd deployment, see [`packaging/systemd/`](packaging/systemd/).

## Quick start

All three transports serve the same HTTP API:

```sh
# Unix domain socket (Linux default)
sudo zeusd serve --socket-path /run/zeusd/zeusd.sock --socket-permissions 666

# TCP (cluster-wide monitoring; required for Python clients on Windows)
sudo zeusd serve --mode tcp --tcp-bind-address 0.0.0.0:4938

# Windows named pipe (Windows default; from an elevated PowerShell)
zeusd serve --pipe-name \\.\pipe\zeusd
```

To let the Zeus Python library auto-detect the daemon, set one of:

```sh
export ZEUSD_SOCK_PATH=/run/zeusd/zeusd.sock     # UDS
export ZEUSD_HOST_PORT=node1:4938                # TCP
```

When set, `NVIDIAGPUs` and `RAPLCPUs` automatically relay privileged calls and CPU/DRAM reads through the daemon.

## Testing

Unit and integration tests run via `cargo test` inside `zeusd/`. CI runs the matrix on every push.

For Windows-specific end-to-end coverage with a real NVIDIA GPU, two helper scripts live in `zeusd/scripts/`:

- `test-windows-gpu.sh` provisions a `g4dn.xlarge` on AWS, builds `zeusd` from a Git ref, runs TCP + named-pipe smoke tests, a PyTorch matmul load with NVML power-limit / locked-clocks round-trips, and an SDDL ACL test that drives privileged NVML writes from an unprivileged client through the elevated daemon. Tears its own resources down via a trap. Pass `-h` for options.
- `cleanup-test-aws.sh` lists or deletes resources tagged `zeusd-test` (used to recover from crashed traps or stale orphans on a shared AWS account). Default mode is list-only; deletion requires an explicit scope (`--tag-value`, `--older-than`, or `--all`). Pass `-h` for the full usage.

Both scripts require `aws` CLI v2 with valid credentials and `jq`. Resources are tagged with a unique per-run value (`zeusd-test-<utc-second>-<pid>-<6 hex chars>`), so concurrent runs by multiple devs on the same AWS account don't collide and a scoped cleanup affects only its own run.

## Documentation

API groups (`gpu-control`, `gpu-read`, `cpu-read`), JWT authentication, Windows-specific behavior, the full HTTP API reference, troubleshooting, and Python client integration are all covered at <https://ml.energy/zeus/zeusd/>.
