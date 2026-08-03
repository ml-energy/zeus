# systemd packaging for `zeusd`

Two files plus this README:

- `zeusd.service` -- the unit file.
- `zeusd.defaults` -- example `EnvironmentFile` for `/etc/default/zeusd`.

## Install

`zeusd` must be on `PATH` at `/usr/local/bin/zeusd` (e.g., `cargo install zeusd`). If installed elsewhere, see *Customize* below.

```sh
sudo install -m 0644 zeusd.service /etc/systemd/system/zeusd.service
sudo install -m 0644 zeusd.defaults /etc/default/zeusd
sudo systemctl daemon-reload
sudo systemctl enable --now zeusd
```

Default config: UDS mode on `/run/zeusd/zeusd.sock`, all API groups enabled.

## Customize

Two layers of override, both survive package upgrades:

- **CLI args** -- edit `/etc/default/zeusd` and set `ZEUSD_ARGS=...`, then `sudo systemctl restart zeusd`. The same file is loaded as an `EnvironmentFile`, so runtime environment variables like `ROCM_PATH` or `AMDSMI_LIB_DIR` (AMD SMI library discovery) also go there.
- **Unit directives** -- `sudo systemctl edit zeusd` opens a drop-in at `/etc/systemd/system/zeusd.service.d/override.conf`. Use this to override `ExecStart=` (e.g., if `zeusd` lives in `/opt/zeusd/bin/`) or to relax a hardening directive. Do not edit the upstream unit in place.

Example drop-in for a non-standard binary path:

```ini
[Service]
ExecStart=
ExecStart=/opt/zeusd/bin/zeusd serve $ZEUSD_ARGS
```

The empty `ExecStart=` line clears the inherited value before redefining it; systemd requires this for `ExecStart`.

## Additional hardening

The unit ships with `ProtectKernelTunables=false` because AMD GPU control (clock limits, power cap, performance level) writes to the amdgpu driver's sysfs files, which that directive would mount read-only.
NVIDIA GPU control goes through `/dev/nvidia*` ioctls and is unaffected, so deployments that never use AMD GPU control can harden further with a drop-in (`sudo systemctl edit zeusd`):

```ini
[Service]
ProtectKernelTunables=true
```

## Verifying

```sh
systemctl status zeusd
journalctl -u zeusd -f
systemd-analyze security zeusd.service
```

With systemd 252, `zeusd.service`'s exposure level score is 3.5 OK; the exact number varies across systemd versions.
