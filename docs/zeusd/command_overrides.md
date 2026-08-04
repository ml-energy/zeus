# GPU Command Overrides

Some machines never grant root.
On shared HPC clusters, for example, the closest thing to privileged GPU control is often a set of admin-blessed commands, such as sudoers-whitelisted scripts wrapping the vendor CLI.
The native `gpu-control` path cannot run there, because setting power limits and clock ranges through NVML or AMD SMI requires root.

`zeusd serve --gpu-command-overrides PATH` closes this gap by replacing selected privileged GPU writes with commands from a TOML file.
The daemon runs unprivileged, the configured commands carry the privilege, and clients keep using the same HTTP API and Zeus Python integrations unchanged.

```toml
[set_power_limit]
commands = ["sudo /usr/local/bin/set_powercap.sh -value {power_limit_w} -gpu {gpu_id}"]
error_pattern = "(?i)error|fail|cannot|unable|not supported"

[set_gpu_locked_clocks]
commands = [
  "sudo /usr/local/bin/set_gpu_clockfreq.sh -clock sclk -limit min -value 500 -gpu {gpu_id}",
  "sudo /usr/local/bin/set_gpu_clockfreq.sh -clock sclk -limit max -value {max_clock_mhz} -gpu {gpu_id}",
  "sudo /usr/local/bin/set_gpu_clockfreq.sh -clock sclk -limit min -value {min_clock_mhz} -gpu {gpu_id}",
]
error_pattern = "(?i)error|fail|cannot|unable|not supported"

[reset_locked_clocks]
commands = ["sudo /usr/local/bin/reset_gpu_clocks.sh"]
error_pattern = "(?i)error|fail|cannot|unable|not supported"
```

Any subset of the seven operations may be configured.
Each operation requires a nonempty `commands` array and accepts an optional `timeout_s`, which defaults to 60 seconds and applies to each command in the array separately.

| Operations | Valid placeholders |
|---|---|
| All operations | `{gpu_id}` |
| `set_power_limit` | `{power_limit_mw}`, `{power_limit_w}` |
| `set_gpu_locked_clocks`, `set_mem_locked_clocks` | `{min_clock_mhz}`, `{max_clock_mhz}` |
| `set_persistence_mode` | `{enabled}` (`1` or `0`) |
| `reset_gpu_locked_clocks`, `reset_mem_locked_clocks`, `reset_locked_clocks` | Only `{gpu_id}` |

Commands are word-split and executed directly, without a shell.
Use a wrapper script when pipes, redirection, or other shell behavior is needed.
Commands run as the zeusd user, so the command must perform any required privilege escalation, such as passwordless `sudo`; no TTY is available.

An `error_pattern` is matched against combined stdout and stderr, and a match fails the operation even after a zero exit status; `amd-smi`, for example, can exit zero when the kernel rejects a write.
Commands run sequentially and stop at the first failure; commands that already ran are not rolled back.

A command without `{gpu_id}` affects whatever scope the command implements; the reset script above resets every GPU in the node.
For CLIs that set one clock bound per call, a plain min-then-max pair fails whenever the new minimum is above the currently applied maximum; the example's three-command sequence (minimum to the hardware floor, then maximum, then minimum) works from any starting range.

When at least one override is configured, `gpu-control` no longer requires zeusd itself to run as root.
Without root, control operations that lack an override use the native driver path and will fail with driver permission errors.

Because the file decides which commands zeusd executes, zeusd refuses to start on Unix if it is world-writable or owned by neither root nor the user running zeusd; keep its containing directory equally protected.
