# Deployment

Choose the features you need, then grant their combined requirements using the deployment method below.
Kernel modules and firmware support are **prerequisites**; file access, Linux capabilities, and container policies are **permissions**.
`zeusd` does not load modules or change host security settings.

## Feature requirements and permissions

Enable HTTP API groups with `--enable gpu-read,gpu-control,cpu-read,cpu-control`, omitting groups you do not need.
Linux enables all four by default; Windows supports only the GPU groups through NVML.
Enabling a group exposes its routes but does not grant device permissions.
Unavailable operations report their missing requirements; independent features remain usable.

CPU features use the Running Average Power Limit (RAPL) interface, Intel model-specific registers (MSRs), or AMD's Host System Management Port (HSMP).
Each interface has separate access requirements.
Linux exposes device attributes through *sysfs*, normally mounted at `/sys`.
CPU resets restore a *baseline*, a saved set of CPU power limits and time windows.

| Feature | API group | Host prerequisites | Daemon permissions |
|---|---|---|---|
| NVIDIA GPU monitoring and limit queries | `gpu-read` | NVIDIA driver and NVIDIA Management Library (NVML) | Access to NVIDIA device nodes; no control capability needed |
| NVIDIA GPU power limits, clock limits, and persistence mode | `gpu-control` | Same as monitoring; operation supported by the GPU | Root/admin access required by NVML; Linux containers also need `CAP_SYS_ADMIN` |
| AMD GPU monitoring and limit queries | `gpu-read` | `amdgpu` driver and a [compatible AMD SMI library](index.md#amd-gpu) | Access to GPU devices and readable GPU sysfs files |
| AMD GPU power and clock limits | `gpu-control` | Same as monitoring; operation supported by the GPU | Root access required by AMD SMI, writable GPU sysfs files, and a security policy allowing those writes |
| CPU/DRAM energy and power monitoring | `cpu-read` | Linux RAPL powercap interface | Read access to `energy_uj` and zone metadata; energy files are normally root-only |
| Intel current CPU limits and time-window queries | `cpu-read` | Intel RAPL powercap constraints | Read access to constraint files; no MSR access needed |
| Intel CPU power-limit changes | `cpu-control` | Intel RAPL powercap constraints | Read/write access to `constraint_*_power_limit_uw`; no MSR access needed |
| Intel hardware power-range queries | `cpu-read` | Supported Intel x86-64 CPU and `msr` driver | MSR device read access and `CAP_SYS_RAWIO` |
| Intel CPU time-window changes | `cpu-control` | Supported Intel x86-64 CPU and `msr` driver | MSR device read/write access, `CAP_SYS_RAWIO`, and kernel/firmware policy permitting MSR writes |
| AMD EPYC current and maximum socket-limit queries | `cpu-read` | Firmware support for the `amd_hsmp` driver; RAPL zones for package discovery | Read access to `/dev/hsmp` |
| AMD EPYC socket power-limit changes | `cpu-control` | Same as socket-limit queries | Read/write access to `/dev/hsmp` |
| CPU resets that retain the original baseline across daemon restarts | `cpu-control` | Persistent baseline storage and the host boot identifier | Read/write access to `/var/zeusd` or the configured baseline directory; restoring each setting also needs its control permissions |

The baseline's [storage requirements](#cpu-reset-baseline) are independent of socket transport and device access.
GPU resets use the GPU backend's defaults and do not use this file.

File access can come from ownership, group permissions, or an administrator-managed policy.
Linux capabilities grant additional privileges: `CAP_SYS_RAWIO` permits MSR device access, while `CAP_SYS_ADMIN` permits NVIDIA control operations.
Neither makes a read-only filesystem writable.
An unprivileged process can read root-only RAPL energy files with `CAP_DAC_READ_SEARCH`, but that capability also bypasses read permissions elsewhere.
Root inside a container still needs the device mounts and capabilities listed below.

If privileged GPU control is available only through approved commands, configure [GPU command overrides](command_overrides.md).
The commands' permissions determine whether those operations succeed.

### CPU driver prerequisites

Linux exposes RAPL energy counters and Intel power constraints through [powercap sysfs files](https://docs.kernel.org/power/powercap/powercap.html).
Check that the host exposes package zones under `/sys/class/powercap/intel-rapl` before enabling CPU groups.
AMD RAPL provides energy counters; AMD EPYC socket limits use HSMP.
Load the relevant optional driver on the **host**, before starting `zeusd`:

```sh
# Intel hardware-range queries and time-window control
sudo modprobe msr

# AMD EPYC socket power-limit queries and control
sudo modprobe amd_hsmp
```

If `amd_hsmp` reports that HSMP is disabled, enable it in the firmware settings.
The [Linux HSMP driver](https://kernel.org/doc/html/v6.2/x86/amd_hsmp.html) distinguishes read access for queries from write access for control.
Missing HSMP access does not prevent RAPL energy monitoring.

Intel MSRs provide power ranges and precise time-window control.
Expose `/dev/cpu/<core>/msr` for the lowest-numbered online core in each package, or each die for per-die RAPL zones.
CPU topology under `/sys/devices/system/cpu` must also be readable.
Time-window control supports package `long_term` and `short_term` constraints with exponential encoding; Silvermont and Airmont layouts are unsupported.
Power-limit writes use RAPL sysfs and verify readback, allowing hardware rounding of less than 1 W.
Time-window writes use MSRs so resets can restore fractional encodings precisely.

!!! warning "MSR writes affect the host kernel"

    Userspace MSR writes taint the Linux kernel until reboot, including writes that restore earlier settings.
    Kernel lockdown, `msr.allow_writes=off`, or a firmware register lock can reject writes even when reads work.
    Without MSR access, energy monitoring, current-limit queries, and sysfs power-limit changes remain available.
    Hardware-range queries and time-window changes return an error explaining the required access.

### CPU reset baseline

`zeusd` changes CPU settings only through control requests.
Starting, stopping, or restarting the daemon does not reset CPU settings.
Call [`POST /cpu/reset_power_limit`](api.md#cpu) or [`ZeusdClient.reset_cpu_power_limit`][zeus.utils.zeusd.ZeusdClient.reset_cpu_power_limit] to restore the baseline explicitly.
A failed write does not stop restoration of the remaining settings, and the request reports the errors.

With `cpu-control` enabled, `--cpu-power-limit-baseline-path` defaults to `/var/zeusd/cpu_power_limit_baseline.json`.
The actual filename includes the host boot identifier from `/proc/sys/kernel/random/boot_id`: `cpu_power_limit_baseline.<boot_id>.json`.
Within one host boot, daemon restarts reuse the saved baseline instead of recording a limit that an application has already changed.
After a host reboot, `zeusd` records a new baseline from the current settings.
Files from older boots are not reused and can be removed.
This baseline captures settings at the first daemon start with CPU control, not firmware defaults or settings from before that start.

Choose a different location with `--cpu-power-limit-baseline-path PATH`.
The directory must be writable and survive daemon or container replacement; a container's writable layer does not survive replacement.
The filesystem must support hard links (multiple filenames for one file), which allow concurrent starts to publish a baseline without overwriting one another.
An inaccessible directory or invalid baseline produces a startup error instead of silently selecting a different baseline.
For a baseline kept only in memory, pass `--no-persistent-cpu-power-limit-baseline` explicitly.
That mode records current settings on every daemon start, so it cannot restore a baseline from an earlier process.

## Deployment methods

The examples below target Linux and assume the [host prerequisites](#feature-requirements-and-permissions) are installed.
Clients connect through a Unix domain socket (UDS) or a TCP port.
For Windows, run native NVML deployments from an elevated shell for GPU control and use `--mode tcp` for Python clients.

=== "Native"

    Install [the binary](index.md#install), then select the API groups and transport.
    GPU monitoring can run without root when the account can access the GPU devices:

    ```sh
    zeusd serve --enable gpu-read --mode tcp --tcp-bind-address 127.0.0.1:4938
    ```

    A root process can use the host's device permissions for CPU monitoring and control:

    ```sh
    sudo install -d -m 0755 /var/zeusd
    sudo "$(command -v zeusd)" serve --enable cpu-read,cpu-control \
        --mode tcp --tcp-bind-address 127.0.0.1:4938
    ```

    Add GPU groups as needed.
    For a restricted service account, grant only the file access and capabilities for its selected features.
    CPU monitoring does not require writable sysfs, MSR access, HSMP access, or baseline storage.
    CPU control without persistent storage requires `--no-persistent-cpu-power-limit-baseline`.

    For UDS, use `--socket-path /run/zeusd/zeusd.sock` instead of the TCP arguments.
    The daemon needs a writable parent directory, and clients need write permission on the socket.
    The default socket mode is `666`; use `--socket-permissions 660` and `--socket-gid GROUP_ID` to restrict clients to a group.
    Assigning a different socket owner or group can require `CAP_CHOWN`.

=== "systemd"

    The [packaging directory](https://github.com/ml-energy/zeus/tree/master/zeusd/packaging/systemd) includes a root service and an environment file for daemon arguments.
    With the binary installed at `/usr/local/bin/zeusd`, install these files from the repository:

    ```sh
    cd zeusd/packaging/systemd
    sudo install -m 0644 zeusd.service /etc/systemd/system/zeusd.service
    sudo install -m 0644 zeusd.defaults /etc/default/zeusd
    sudo systemctl daemon-reload
    sudo systemctl enable --now zeusd
    ```

    Set `ZEUSD_ARGS` in `/etc/default/zeusd` to select API groups, transport, and baseline mode.
    The unit creates `/run/zeusd` for sockets and permits persistent baseline storage under `/var/zeusd`.
    Load optional host drivers before starting the service; `ProtectKernelModules=true` prevents the service from loading them itself.

    Use `sudo systemctl edit zeusd` to restrict the unit for your selected features:

    | Feature | Unit requirement |
    |---|---|
    | NVIDIA control | Keep `CAP_SYS_ADMIN` in `CapabilityBoundingSet` |
    | Intel MSR queries or control | Keep `CAP_SYS_RAWIO`; permit the required MSR device access |
    | RAPL or AMD GPU sysfs control | Keep `ProtectKernelTunables=false`; allow writes through any added path restrictions |
    | Persistent CPU reset baseline | Permit writes to the configured baseline directory under `ProtectSystem=strict` |
    | Custom socket ownership | Keep `CAP_CHOWN` when changing ownership |

    A monitoring-only deployment can remove control capabilities and set `ProtectKernelTunables=true`.
    For example, with `ZEUSD_ARGS="--enable gpu-read,cpu-read"`, a drop-in for the root service can use:

    ```ini
    [Service]
    CapabilityBoundingSet=
    ProtectKernelTunables=true
    ```

    The empty `CapabilityBoundingSet=` clears the capabilities from the packaged unit.
    When retaining selected capabilities, clear the list first, then add a second assignment with the required capabilities.
    If using `User=` for a service account, grant its device/file permissions separately; `CapabilityBoundingSet` alone does not grant capabilities to that account.
    Use `AmbientCapabilities=` for any capabilities that account requires.
    Read logs with `journalctl -u zeusd -f` after restarting the service.

=== "Docker"

    [Images](https://hub.docker.com/r/mlenergy/zeusd) are available for amd64 and arm64.
    Release tags and `latest` track releases; `master` tracks the master branch.
    AMD SMI is bundled on amd64; NVIDIA deployments need the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html).
    These examples assume the Docker daemon runs as root and does not remap container user IDs.

    Start with monitoring for your GPU vendor:

    ```sh
    # NVIDIA
    docker run -d --gpus all -p 127.0.0.1:4938:4938 \
        mlenergy/zeusd serve --enable gpu-read \
        --mode tcp --tcp-bind-address 0.0.0.0:4938

    # AMD
    docker run -d --device /dev/dri --device /dev/kfd \
        -p 127.0.0.1:4938:4938 \
        mlenergy/zeusd serve --enable gpu-read \
        --mode tcp --tcp-bind-address 0.0.0.0:4938
    ```

    Add these arguments **before the image name** for additional features, and enable the corresponding API groups after `serve`:

    | Feature | Docker arguments |
    |---|---|
    | NVIDIA control | `--cap-add SYS_ADMIN` |
    | CPU monitoring or Intel limit queries | `-v /sys/class/powercap:/zeus_sys/class/powercap:ro -v /sys/devices/virtual/powercap:/zeus_sys/devices/virtual/powercap:ro` |
    | Intel power-limit changes | Use the two powercap mounts above with `:rw` instead of `:ro` |
    | Intel hardware-range queries | `--cap-add SYS_RAWIO --device /dev/cpu/CORE/msr:/dev/cpu/CORE/msr:r` for each required core |
    | Intel time-window changes | Use the same MSR arguments with `:rw` instead of `:r` |
    | AMD EPYC socket-limit queries | `--device /dev/hsmp:/dev/hsmp:r`, plus the CPU monitoring mounts for package discovery |
    | AMD EPYC socket-limit changes | Use the same HSMP argument with `:rw` instead of `:r` |
    | Persistent CPU reset baseline | `-v /var/zeusd:/var/zeusd` and expose the host boot identifier as described below |
    | Share a UDS socket | `-v /run/zeusd:/run/zeusd`; use `--socket-path /run/zeusd/zeusd.sock` instead of TCP arguments |

    Replace `CORE` with the representative core number described under [CPU driver prerequisites](#cpu-driver-prerequisites).
    The two powercap mounts preserve sysfs symlink targets under `/zeus_sys`, where `zeusd` looks when the normal sysfs path is masked.
    Container security policies must also permit the selected sysfs writes; a writable bind mount alone cannot override AppArmor or SELinux restrictions.

    For example, Intel power-limit control with resets that survive container replacement uses:

    ```sh
    sudo install -d -m 0755 /var/zeusd
    docker run -d -p 127.0.0.1:4938:4938 \
        -v /sys/class/powercap:/zeus_sys/class/powercap:rw \
        -v /sys/devices/virtual/powercap:/zeus_sys/devices/virtual/powercap:rw \
        -v /var/zeusd:/var/zeusd \
        --mount type=bind,src=/proc/sys/kernel/random/boot_id,dst=/proc/sys/kernel/random/boot_id,readonly \
        mlenergy/zeusd serve --enable cpu-read,cpu-control \
        --mode tcp --tcp-bind-address 0.0.0.0:4938
    ```

    Binding the host boot identifier ensures baseline reuse follows host reboots even when a runtime supplies a container-specific identifier.
    Omit the storage and boot-identifier mounts only when using `--no-persistent-cpu-power-limit-baseline` or disabling `cpu-control`.
    TCP deployments need no `/run/zeusd` mount.

    AMD GPU control additionally needs writable amdgpu sysfs files.
    A broad configuration uses `-v /sys:/sys:rw` with a policy that permits those writes; a narrower deployment can bind only the required GPU sysfs paths.
    Docker's default AppArmor policy blocks sysfs writes; supply a custom policy, or use `--security-opt apparmor=unconfined` to remove that protection.
    SELinux deployments need a policy allowing device and sysfs access; `--security-opt label=disable` disables container labeling when that is acceptable to your deployment.
    These policy changes apply independently of `--cap-add`.

    To use a host AMD SMI library, mount its installation and set `AMDSMI_LIB_DIR`, for example `-v /opt/rocm-7.2.0:/opt/rocm-7.2.0:ro -e AMDSMI_LIB_DIR=/opt/rocm-7.2.0/lib`.

## Client access and troubleshooting

Device permissions govern what the daemon can do; socket access and [JWT scopes](index.md#authentication-optional) govern who can request it.
For remote TCP access, select the intended bind address and configure authentication before exposing the port.
`/discover` lists detected devices and enabled API groups; it does not guarantee permission for every operation in a group.
For GPU writes, use `block=true` to receive execution errors in the HTTP response; nonblocking writes report execution failures in daemon logs.

??? tip "Diagnosing unavailable features"

    Check both the HTTP error and the daemon logs for the failing path, device, or capability.
    A missing device usually indicates a driver, firmware, or container-device requirement; `Permission denied` indicates denied access, and `Read-only file system` indicates a mount restriction.
    After changing modules or device visibility, restart the daemon so it can discover the new interfaces.
    A baseline mismatch requires the original package/constraint interfaces to be restored, or an explicit decision to record a new baseline from the current settings.
    Do not remove a baseline while a daemon is using it.
