# systemd packaging for `zeusd`

Two files plus this README:

- `zeusd.service` -- the unit file.
- `zeusd.defaults` -- example `EnvironmentFile` for `/etc/default/zeusd`.

## Install

The unit expects `zeusd` at `/usr/local/bin/zeusd`.
If installed elsewhere, see *Customize* below.

```sh
sudo install -m 0644 zeusd.service /etc/systemd/system/zeusd.service
sudo install -m 0644 zeusd.defaults /etc/default/zeusd
sudo systemctl daemon-reload
sudo systemctl enable --now zeusd
```

Default config: UDS mode on `/run/zeusd/zeusd.sock`, all API groups enabled.

## Original CPU power limits

By default, `POST /cpu/reset_power_limit` restores the original CPU power limits, which `zeusd` records at its first start with CPU control in the current boot.
The record is a file in `/var/zeusd`, whose name contains the host boot ID, so it survives service restarts and a reboot starts a new one.
The unit creates `/var/zeusd` and lists it in `ReadWritePaths=`, which `ProtectSystem=strict` otherwise makes read-only.
To keep the record in memory instead, add `--no-persistent-original-cpu-power-limit` to `ZEUSD_ARGS`; each restart then records the limits it finds at that time.

## Customize

Two layers of override, both survive package upgrades:

- **CLI args** -- edit `/etc/default/zeusd` and set `ZEUSD_ARGS=...`, then `sudo systemctl restart zeusd`.
  The same file is loaded as an `EnvironmentFile`, so runtime environment variables like `ROCM_PATH` or `AMDSMI_LIB_DIR` (AMD SMI library discovery) also go there.
- **Unit directives** -- `sudo systemctl edit zeusd` opens a drop-in at `/etc/systemd/system/zeusd.service.d/override.conf`.
  Use this to override `ExecStart=` (e.g., if `zeusd` lives in `/opt/zeusd/bin/`) or to relax a hardening directive.
  Do not edit the upstream unit in place.

Example drop-in for a non-standard binary path:

```ini
[Service]
ExecStart=
ExecStart=/opt/zeusd/bin/zeusd serve $ZEUSD_ARGS
```

The empty `ExecStart=` line clears the inherited value before redefining it; systemd requires this for `ExecStart`.

## Additional hardening

The unit ships with `ProtectKernelTunables=false` because AMD GPU control and RAPL CPU power-limit control write sysfs files, which that directive would mount read-only.
Deployments that use neither can harden further with a drop-in (`sudo systemctl edit zeusd`):

```ini
[Service]
ProtectKernelTunables=true
```

For the drivers, files, and capabilities each feature needs, see [Feature requirements and permissions](https://ml.energy/zeus/zeusd/deployment/#feature-requirements-and-permissions).

## Verifying

```sh
systemctl status zeusd
journalctl -u zeusd -f
systemd-analyze security zeusd.service
```
