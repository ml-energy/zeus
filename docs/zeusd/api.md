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

## Tracked power limit operations

An operation is a power limit request with a retained result for each selected device.
It continues after the submitting HTTP connection closes.
These endpoints require `gpu-control` or `cpu-control`, including status and cancellation requests.

| Method | GPU path | CPU path |
|---|---|---|
| `POST` | `/gpu/set_power_limit/operations` | `/cpu/set_power_limit/operations` |
| `GET` | `/gpu/set_power_limit/operations/{id}` | `/cpu/set_power_limit/operations/{id}` |
| `DELETE` | `/gpu/set_power_limit/operations/{id}` | `/cpu/set_power_limit/operations/{id}` |

POST requires `gpu_ids` or `cpu_ids`, a positive `power_limit_mw`, and a positive `start_within_ms`.
CPU requests also require `constraint`: `long_term`, `short_term`, `peak_power`, or `socket`.
`start_within_ms` limits queue waiting from server admission until the device executor begins backend execution.
It has no default and does not limit execution time.
There is no `block` parameter or client-supplied ID.
Duplicate device IDs, nonexistent devices, unknown fields, and unrepresentable values are rejected before any selected device is enqueued.
Checks requiring hardware access run during execution and can fail independently on each device.

```text
POST /gpu/set_power_limit/operations?gpu_ids=0,1&power_limit_mw=250000&start_within_ms=1000
POST /cpu/set_power_limit/operations?cpu_ids=0&constraint=long_term&power_limit_mw=150000&start_within_ms=1000
```

Accepted POST requests return `202`, a `Location` header containing the status URL, and a snapshot.
GET and DELETE return the current snapshot with `200`.
Operation responses use `Cache-Control: no-store`.

```json
{
  "id": "31415926-5358-4979-8323-846264338327:1",
  "status_url": "/gpu/set_power_limit/operations/31415926-5358-4979-8323-846264338327:1",
  "device_type": "gpu",
  "requested": {"power_limit_mw": 250000},
  "start_within_ms": 1000,
  "complete": false,
  "devices": {
    "0": {"state": "running", "effect": "uncertain", "reason": null, "error": null},
    "1": {"state": "queued", "effect": "unchanged", "reason": null, "error": null}
  }
}
```

`requested` is immutable and includes `constraint` for CPU operations.
`complete` becomes true when every device has a terminal state: `succeeded`, `failed`, or `cancelled`.
Device states and evidence about the command's effect are separate:

| State | Meaning | Effect |
|---|---|---|
| `queued` | Waiting for its device executor | `unchanged` |
| `running` | Backend execution has begun; cancellation is unavailable | `uncertain` |
| `succeeded` | The backend completed the setting command successfully | `applied` |
| `failed` | Backend execution failed | `unchanged`, `restored`, or `uncertain`, according to backend evidence |
| `cancelled` | Execution was prevented | `unchanged` |

DELETE cancels each device that is still queued and retains its result.
Repeated DELETE is safe; running and terminal results are unchanged.
Cancellation and execution compete under one lock, so a cancelled device's setter cannot run later.
`reason` identifies `cancellation_requested`, `deadline_exceeded`, `executor_unavailable`, `executor_panicked`, or `backend_error`; `error` contains a backend message when available.
Queue deadlines remain observable even if an earlier native call is stuck.
An HTTP timeout or an elapsed queue deadline cannot stop an executing native call.

Existing and tracked commands share one ordered executor per device.
Different devices have independent threads, but vendor locks can serialize backend calls across devices.
In particular, older AMD SMI versions require a global lock.
Backend validation and waiting for vendor locks are part of `running` execution.
Background power polling retains separate handles and can overlap command execution where the backend permits it.
Multiple-device operations can partially succeed or partially cancel; they are not atomic across devices.

`unchanged` means this command did not write the setting.
`restored` means the backend verified the previous setting after attempting a write.
RAPL verifies its existing rounding and restoration behavior; tracking adds no universal GPU readback.
Generic vendor errors after execution begins are `uncertain`.
`applied` acknowledges the setting command; it does not guarantee workload power consumption or prevent later changes by another actor.

IDs contain a random server startup UUID and a checked increasing counter; clients should treat the complete ID as opaque.
Results are held in memory and are never replayed after restart.
A different startup UUID or an expired or evicted result returns `410` with `error.code = "operation_unavailable"` and `effect = "uncertain"`.
A well-formed, unknown current-startup ID returns `404` with `error.code = "operation_not_found"`.
If the POST response is lost, the client may never learn the ID, and retrying can submit another operation.
The API does not guarantee exactly-once submission across blind retries.

The shared registry permits 1,024 live operations, 4,096 completed operations, and 16,384 retained device results.
Completed results expire after one hour and can be evicted earlier for count or device-result capacity.
Queued or running results are never evicted to admit another operation.
Each device also permits 1,024 outstanding tracked queue entries, including cancelled entries waiting behind a blocked command.
Admission fails with `503` and `error.code = "capacity_exhausted"` when these bounds prevent reserving every result and queue entry.
Rejected admission enqueues no device work.
Error messages retain at most 2,048 bytes plus a truncation marker, bounding message content to approximately 32 MiB across all device results, plus allocation overhead.
Existing untracked queues retain their previous admission behavior and do not allocate retained operation results.
