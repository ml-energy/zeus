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

- **CLI args** -- edit `/etc/default/zeusd` and set `ZEUSD_ARGS=...`, then `sudo systemctl restart zeusd`.
- **Unit directives** -- `sudo systemctl edit zeusd` opens a drop-in at `/etc/systemd/system/zeusd.service.d/override.conf`. Use this to override `ExecStart=` (e.g., if `zeusd` lives in `/opt/zeusd/bin/`) or to relax a hardening directive. Do not edit the upstream unit in place.

Example drop-in for a non-standard binary path:

```ini
[Service]
ExecStart=
ExecStart=/opt/zeusd/bin/zeusd serve $ZEUSD_ARGS
```

The empty `ExecStart=` line clears the inherited value before redefining it; systemd requires this for `ExecStart`.

## AMD GPUs

AMD GPU control writes clock limits, the power cap, and the performance level through the amdgpu driver's sysfs files.
The unit therefore ships with `ProtectKernelTunables=false`; setting it to true would mount `/sys` read-only and block those writes.
NVIDIA control uses `/dev/nvidia*` ioctls and does not care, so NVIDIA-only nodes can re-enable the directive with a drop-in (`sudo systemctl edit zeusd`):

```ini
[Service]
ProtectKernelTunables=true
```

Environment variables such as `ROCM_PATH` or `AMDSMI_LIB_DIR` can be set in `/etc/default/zeusd` because the unit loads it as an `EnvironmentFile`.

## Verifying

```sh
systemctl status zeusd
journalctl -u zeusd -f
systemd-analyze security zeusd.service
```

At the time of writing, `zeusd.service`'s systemd exposure level score is 3.5 OK.
