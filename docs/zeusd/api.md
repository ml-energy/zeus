# HTTP API Reference

The API is the same regardless of transport. Paths shown below are server-relative; prefix with `http://<host>:<port>` over TCP, the UDS socket over UDS, or the named pipe on Windows.

Status codes: `200` success; `400` bad input or unsupported op (e.g., persistence-mode off on Windows); `401` missing/invalid token; `403` insufficient token scope, device permissions, kernel write policy, or a CPU power limit locked by the BIOS; `404` disabled API group or `/auth/*` on a no-auth daemon; `500` daemon-side failure, e.g., an unexpected driver error or a failed [override command](command_overrides.md); `503` unavailable MSR driver or unsupported MSR time-window layout.
Per-device write calls aggregate per-device errors into `{"errors": {"<device_id>": "<message>"}}` with the worst per-device status.

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
  "enabled_api_groups": ["gpu-control", "gpu-read", "cpu-read", "cpu-control"],
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

The write endpoints in this table also take `block` (bool): `true` waits for completion and reports per-GPU execution errors; `false` dispatches non-blocking and only reports command send errors.
For retained asynchronous power limit results, use [tracked power limit operations](#tracked-power-limit-operations).

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

All endpoints are under `/cpu` (Linux only). `cpu_ids` is a comma-separated list of RAPL package indices (the `N` in `/sys/class/powercap/intel-rapl/intel-rapl:N/`, not core or hyperthread IDs); optional on `GET` endpoints (omit to read all CPUs) and required on `POST` endpoints.

| Method | Path | Extra params / notes |
|---|---|---|
| `GET` | `/cpu/get_cumulative_energy` | `cpu` (bool) and `dram` (bool), both required |
| `GET` | `/cpu/get_power` | one-shot snapshot |
| `GET` | `/cpu/stream_power` | SSE stream |
| `GET` | `/cpu/get_power_limit` | power limit constraints |
| `GET` | `/cpu/get_power_limit_constraints` | ranges the hardware reports for power limits |
| `POST` | `/cpu/set_power_limit` | `constraint`, `power_limit_mw` |
| `POST` | `/cpu/set_power_limit_time_window` | `constraint`, `time_window_us` |
| `POST` | `/cpu/reset_power_limit` | |

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

`get_power_limit` returns the power limits of each CPU package (`cpu`) and its DRAM zone (`dram`, `null` if absent):

```json
{
  "0": {
    "cpu": {
      "enabled": true,
      "constraints": [
        {"name": "long_term", "power_limit_mw": 205000, "max_power_mw": 205000, "time_window_us": 999424},
        {"name": "short_term", "power_limit_mw": 246000, "max_power_mw": 780000, "time_window_us": 999424},
        {"name": "peak_power", "power_limit_mw": 300000, "max_power_mw": 1560000, "time_window_us": null}
      ]
    },
    "dram": {
      "enabled": false,
      "constraints": [
        {"name": "long_term", "power_limit_mw": 0, "max_power_mw": 121000, "time_window_us": 976}
      ]
    }
  }
}
```

Each field mirrors a file in the zone's powercap sysfs directory: `enabled` is `enabled`, and the constraint at array position `K` comes from the `constraint_K_*` files, with power converted to milliwatts.
`max_power_mw` and `time_window_us` are `null` when the kernel has no value for the attribute (its sysfs read fails with `ENODATA`); kernels 6.5 and later have no time window for `peak_power`.
`enabled` reflects the zone's RAPL `long_term` limit only, and the kernel reports `false` when that limit is disabled, locked by the BIOS, or its enable bit could not be read.
The daemon never changes enable bits; `enabled` does not report whether every constraint is active.

AMD CPUs expose no RAPL power limits.
On AMD EPYC CPUs with the `amd_hsmp` kernel module loaded (which creates `/dev/hsmp`), the package zone instead has a constraint named `socket`: the socket power limit enforced by the CPU's firmware, with its maximum in `max_power_mw` and `time_window_us` always `null`.
`constraints` is empty when neither is available.

`set_power_limit` and `set_power_limit_time_window` change one constraint of each listed CPU's package zone, named as in `get_power_limit`.
Both reject a constraint without a recorded [original setting](deployment.md#original-cpu-power-limits).
`set_power_limit` rejects a constraint the zone does not have, `0`, a `socket` limit above its `max_power_mw` (the firmware would clamp it), and a RAPL limit too large for the CPU's register.
For a RAPL limit that does not fit, Zeusd attempts to restore the previous limit and reports whether restoration was verified.
The hardware can round a RAPL limit down by less than 1 W.
RAPL limits are not bounded by `max_power_mw`; for `long_term` it is the CPU's thermal design power (TDP), which the hardware allows exceeding.
Neither interface reports the lowest power a CPU can hold under load, so a cap below it is accepted but not met.
`set_power_limit_time_window` requires [MSR read/write access](deployment.md#feature-requirements-and-permissions) on supported Intel x86-64 packages and changes only the selected `long_term` or `short_term` window.
Before writing, it rejects constraints without a time window, zero windows, and values above the largest supported window.
The maximum excludes encodings that older Linux kernels cannot read correctly; errors report the maximum for the CPU.
It rounds down to an encodable value, with a minimum of one hardware time unit, and verifies the stored encoding.
MSR writes taint the kernel until reboot; power-limit writes use sysfs and require no MSR access.
Each listed CPU is changed independently, so CPUs that succeed keep the new value even when others fail.

`get_power_limit_constraints` returns the ranges each CPU package reports for its power limits:

```json
{
  "0": {
    "rapl": {
      "thermal_spec_power_mw": 205000,
      "min_power_mw": 113000,
      "max_power_mw": 780000,
      "max_time_window_us": 31981568,
      "power_limit_register_max_mw": 4095875
    },
    "hsmp": null
  }
}
```

`rapl` comes from Intel's `MSR_PKG_POWER_INFO` register and is `null` when the zone has no RAPL constraints or is not a CPU package, such as `psys` (platform power).
`thermal_spec_power_mw` is the TDP; `min_power_mw`, `max_power_mw`, and `max_time_window_us` are the ranges Intel documents as allowed; and `power_limit_register_max_mw` is the largest `long_term` or `short_term` power limit the register can hold.
Reading the register requires [MSR read access](deployment.md#feature-requirements-and-permissions).
`hsmp` is `null` when the package zone has no `socket` constraint, and its `max_power_mw` is the largest socket limit the firmware applies.
The hardware does not enforce the documented ranges: it can accept limits outside them, and whether it holds a limit depends on the load.

`reset_power_limit` explicitly restores the [original settings](deployment.md#original-cpu-power-limits) of every package zone constraint: the power limit and time window at the first daemon start in the host boot that has CPU control and finds CPU packages.
With `--no-persistent-original-cpu-power-limit`, they are the settings at the current daemon's start.
Starting or stopping the daemon does not restore settings.
Values that already match are not written, and a failed write does not stop the remaining ones.
Read failures in RAPL do not block HSMP restoration, and HSMP read failures do not block RAPL restoration.
Changed Intel time windows require MSR write access and are restored exactly, including fractional encodings.
Without that access, reset still attempts power-limit restoration and reports errors for changed windows.

## Tracked operations

A request to change, for instance, a device's power limit, can time out while the hardware command continues.
The client then cannot tell whether the change succeeded.
Tracked operations give each request an ID and retain its results, so clients can check progress after reconnecting instead of submitting the change again.
As of now, tracked operations are supported only for power limit changes.

| Method | GPU path | CPU path |
|---|---|---|
| `POST` | `/gpu/set_power_limit/operations/submit` | `/cpu/set_power_limit/operations/submit` |
| `POST` | `/gpu/set_power_limit/operations/prepare` | `/cpu/set_power_limit/operations/prepare` |
| `POST` | `/gpu/set_power_limit/operations/{id}/activate` | `/cpu/set_power_limit/operations/{id}/activate` |
| `GET` | `/gpu/set_power_limit/operations/{id}` | `/cpu/set_power_limit/operations/{id}` |
| `DELETE` | `/gpu/set_power_limit/operations/{id}` | `/cpu/set_power_limit/operations/{id}` |

All endpoints require the corresponding `gpu-control` or `cpu-control` permission.

### Submit an operation

Both `/submit` and `/prepare` accept `gpu_ids` or `cpu_ids`, a positive `power_limit_mw`, and a positive `start_within_ms`.
`start_within_ms` sets the deadline in milliseconds from preparation, including activation and queue waiting; unstarted work is cancelled when it expires.
CPU requests also require a supported `constraint`, like `/cpu/set_power_limit`.

There are two ways to start an operation:

1. **Submit immediately:** `POST .../operations/submit` prepares and activates the command in one request.
   Execution can start before the reply arrives; if that reply is lost, the client may have no ID to query.
2. **Prepare, then activate:** `POST .../operations/prepare` stores the command without executing it.
   Save the returned ID, then call `POST .../operations/{id}/activate` using only that ID.
   The saved ID lets the client query or retry activation after a lost reply without repeating the write.

For example, the GPU requests below are alternative ways to start the same change:

```text
POST /gpu/set_power_limit/operations/submit?gpu_ids=0,1&power_limit_mw=250000&start_within_ms=1000
POST /gpu/set_power_limit/operations/prepare?gpu_ids=0,1&power_limit_mw=250000&start_within_ms=1000
```

Both return `202 Accepted` with an ID and `status_url`; this is a preparation response:

```json
{
  "id": "31415926-5358-4979-8323-846264338327:1",
  "status_url": "/gpu/set_power_limit/operations/31415926-5358-4979-8323-846264338327:1",
  "device_type": "gpu",
  "requested": {"power_limit_mw": 250000},
  "start_within_ms": 1000,
  "complete": false,
  "devices": {
    "0": {"state": "prepared", "effect": "unchanged", "reason": null, "error": null},
    "1": {"state": "prepared", "effect": "unchanged", "reason": null, "error": null}
  }
}
```

After successful command preparation, activate it to queue the work for execution:

```text
POST /gpu/set_power_limit/operations/31415926-5358-4979-8323-846264338327:1/activate
```

Activation returns `202 Accepted` with the current state; retries neither repeat writes nor extend the deadline.
Regardless of what the activation response is, its state can be queried or retried with that ID.
Queue exhaustion returns `503 Service Unavailable` and leaves the operation prepared.

### Check or cancel an operation

Poll `status_url` with `GET` to check on the operation.
The response schema is the same as that of the initial response above.

The operation is complete when `complete` is `true`.
A prepared device moves from `prepared` to `queued` on activation, then to `running`, and finally `succeeded` or `failed`.
Work cancelled before execution becomes `cancelled`.
Devices can finish independently, so check each result even when `complete` is `true`.

`effect` is `applied` if the write succeeded, or `unchanged` if no write occurred.
It is `restored` if the previous setting was restored and verified, or `uncertain` if the outcome is unknown.
For failures, `reason` and `error` provide details.

Send `DELETE` to `status_url` to cancel prepared or queued work, prevent delayed activation, and retrieve the current results.
Running work continues: neither cancellation nor `start_within_ms` can interrupt a hardware call that has already started.
`GET` and `DELETE` return `200 OK` with the same response structure as `POST`.

Results are held in memory for up to one hour after completion and can be removed earlier to make room for newer results.
A daemon restart also loses them; `410 Gone` means the result is unavailable and the outcome remains unknown.
