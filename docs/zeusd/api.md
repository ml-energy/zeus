# HTTP API Reference

The API is the same regardless of transport. Paths shown below are server-relative; prefix with `http://<host>:<port>` over TCP, the UDS socket over UDS, or the named pipe on Windows.

Status codes: `200` success; `400` bad input or unsupported op (e.g., persistence-mode off on Windows); `401` missing/invalid token; `403` insufficient token scope or NVML `NoPermission`; `404` disabled API group or `/auth/*` on a no-auth daemon. Per-device write calls aggregate per-device errors into `{"errors": {"<device_id>": "<message>"}}` with the worst per-device status.

## `GET /discover`

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
`cumulative_energy_available` states whether the GPU has a trustworthy cumulative energy counter; when false, `GET /gpu/get_cumulative_energy` returns 400 for that GPU (see [Notes on Platforms](index.md#notes-on-platforms)).

## `GET /time`

Daemon-side Unix timestamp in milliseconds. Always available.

```json
{"timestamp_ms": 1762000000000}
```

## `GET /auth/whoami`

Authenticated user's identity and scopes. Requires a bearer token. Returns 404 when auth is disabled.

```json
{"sub": "alice", "scopes": ["gpu-read", "gpu-control"], "exp": 1762864200}
```

`exp` is omitted for tokens issued with `--expires never`.

## GPU

All endpoints are under `/gpu`. `gpu_ids` is a comma-separated list of GPU indices: required on writes; optional on reads (omit to apply to / read all GPUs).

Writes (`POST`) also take `block` (bool): `true` waits for completion and reports per-GPU execution errors; `false` dispatches non-blocking and only reports MPSC send errors.

| Method | Path | Extra params / notes |
|---|---|---|
| `POST` | `/gpu/set_power_limit` | `power_limit_mw` |
| `POST` | `/gpu/set_persistence_mode` | `enabled`; AMD GPUs return 400 because persistence mode is an NVML concept (see [Windows notes](index.md#windows)). |
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

## CPU

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
