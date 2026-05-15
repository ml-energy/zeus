# systemd packaging for `zeusd`

Minimal, idiomatic systemd integration. Two files plus this README:

- `zeusd.service` -- the unit file.
- `zeusd.defaults` -- example `EnvironmentFile` that ships to `/etc/default/zeusd`.

## Install

The binary must already be on `PATH` at `/usr/local/bin/zeusd` (e.g., from `cargo install zeusd`). If installed elsewhere, see *Customize* below.

```sh
sudo install -m 0644 zeusd.service /etc/systemd/system/zeusd.service
sudo install -m 0644 zeusd.defaults /etc/default/zeusd
sudo systemctl daemon-reload
sudo systemctl enable --now zeusd
```

The default config uses UDS mode on `/run/zeusd/zeusd.sock` with all API groups enabled. `RuntimeDirectory=zeusd` creates `/run/zeusd` at start with mode 0755 and removes it on stop.

## Customize

Two layers of override, both survive package upgrades:

- **CLI args** -- edit `/etc/default/zeusd` and set `ZEUSD_ARGS=...`. The shipped file is fully commented and shows common patterns (TCP mode, monitoring-only, JWT auth). Then `sudo systemctl restart zeusd`.
- **Unit-level changes** -- `sudo systemctl edit zeusd` opens a drop-in at `/etc/systemd/system/zeusd.service.d/override.conf`. Use this to override `ExecStart=` (e.g., if `zeusd` lives in `/opt/zeusd/bin/`) or to relax a hardening directive. Do not edit the upstream unit in place.

Example drop-in for a non-standard binary path:

```ini
[Service]
ExecStart=
ExecStart=/opt/zeusd/bin/zeusd serve $ZEUSD_ARGS
```

The empty `ExecStart=` clears the inherited value before redefining it -- systemd requires this for `ExecStart`.

## Verifying

```sh
systemctl status zeusd
journalctl -u zeusd -f
```

The daemon logs to stderr, which systemd captures into the journal.

## Auditing hardening

```sh
systemd-analyze security zeusd.service
```

Expect a score in the "OK" range (around 3.0). Pushing lower would require `MemoryDenyWriteExecute=yes` and `PrivateDevices=yes`, both of which break the daemon: NVML and CUDA runtimes JIT-compile code and need `/dev/nvidia*` device nodes. The active directives (`NoNewPrivileges`, `ProtectSystem=strict`, `ProtectKernel*`, `RestrictAddressFamilies`, `SystemCallFilter=@system-service`, etc.) cover the realistic threat surface.

## Why root

The daemon runs as root because the privileged operations it wraps require two capabilities together:

- `CAP_SYS_ADMIN` -- NVML write operations (set power limit, locked clocks, persistence mode).
- `CAP_DAC_READ_SEARCH` -- reading `/sys/class/powercap/intel-rapl/.../energy_uj`, which is mode 0400 since CVE-2020-8694.

Running as root is the simplest deployment model; the hardening directives in the unit (including a `CapabilityBoundingSet=` that drops every capability except those two) provide defense-in-depth. Operators who want to go further and drop the root identity can override `User=`, add `AmbientCapabilities=CAP_SYS_ADMIN CAP_DAC_READ_SEARCH`, and ship a `sysusers.d` entry for the service user. That is left as an exercise -- the upstream unit does not ship it.

## Platform

Linux only. macOS users deploy via launchd (not provided here). Windows users run the named-pipe transport and can register `zeusd` as a Windows service via tools like NSSM (not provided here).
