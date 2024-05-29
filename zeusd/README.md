# Zeus daemon (`zeusd`)

`zeusd` is a daemon designed to run with admin privileges and expose API endpoints that wrap privileged NVML methods.

## Problem

Energy optimizers in Zeus need to change the GPU's configurations including its power limit or frequency, which requires the Linux security capability `SYS_ADMIN` (which is pretty much `sudo`).
However, it's not a good idea to grant the entire application such strong privileges just to be able to change GPU configurations.

## Solution

`zeusd` runs as a privileged daemon process on the node and provides API endpoints that wrap privileged NVML methods.
Then, unprivileged applications can ask `zeusd` to change the GPU's configuration on their behalf.
To make this as low latency as possible, `zeusd` was written with Rust.

## How to use `zeusd`

First, install `zeusd`:

```sh
cargo install zeusd
```

With the following, `zeusd` will listen to a unix domain socket at `/var/run/zeusd.sock`, which is writable to anyone (since file permission is 666).

```sh
sudo zeusd --socket-path /var/run/zeusd.sock --socket-permissions 666
```

To allow the Zeus Python library to recognize that `zeusd` is available, set:

```sh
export ZEUSD_SOCK_PATH=/var/run/zeusd.sock
```

When Zeus detects `ZEUSD_SOCK_PATH`, it'll automatically instantiate the right GPU backand and relay privileged GPU management method calls to `zeusd`.

### Full help message

```console
$ zeusd --help
The Zeus daemon runs with elevated provileges and communicates with unprivileged Zeus clients to allow them to interact with and control compute devices on the node

Usage: zeusd [OPTIONS]

Options:
      --mode <MODE>
          Operating mode: UDS or TCP
          
          [default: uds]

          Possible values:
          - uds: Unix domain socket
          - tcp: TCP

      --socket-path <SOCKET_PATH>
          [UDS mode] Path to the socket Zeusd will listen on
          
          [default: /var/run/zeusd.sock]

      --socket-permissions <SOCKET_PERMISSIONS>
          [UDS mode] Permissions for the socket file to be created
          
          [default: 666]

      --socket-uid <SOCKET_UID>
          [UDS mode] UID to chown the socket file to

      --socket-gid <SOCKET_GID>
          [UDS mode] GID to chown the socket file to

      --tcp-bind-address <TCP_BIND_ADDRESS>
          [TCP mode] Address to bind to
          
          [default: 127.0.0.1:4938]

      --allow-unprivileged
          If set, Zeusd will not complain about running as non-root

      --num-workers <NUM_WORKERS>
          Number of worker threads to use. Default is the number of logical CPUs

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```
