//! Zeus daemon configuration.

use anyhow::Context;
use clap::{Parser, ValueEnum};

/// The Zeus daemon runs with elevated provileges and communicates with
/// unprivileged Zeus clients to allow them to interact with and control
/// compute devices on the node.
#[derive(Parser, Debug)]
#[command(version)]
pub struct Config {
    /// Operating mode: UDS or TCP.
    #[clap(long, default_value = "uds")]
    pub mode: ConnectionMode,

    /// [UDS mode] Path to the socket Zeusd will listen on.
    #[clap(long, default_value = "/var/run/zeusd.sock")]
    pub socket_path: String,

    /// [UDS mode] Permissions for the socket file to be created.
    #[clap(long, default_value = "666")]
    socket_permissions: String,

    /// [UDS mode] UID to chown the socket file to.
    #[clap(long)]
    pub socket_uid: Option<u32>,

    /// [UDS mode] GID to chown the socket file to.
    #[clap(long)]
    pub socket_gid: Option<u32>,

    /// [TCP mode] Address to bind to.
    #[clap(long, default_value = "127.0.0.1:4938")]
    pub tcp_bind_address: String,

    /// If set, Zeusd will not complain about running as non-root.
    #[clap(long, default_value = "false")]
    pub allow_unprivileged: bool,

    /// Number of worker threads to use. Default is the number of logical CPUs.
    #[clap(long)]
    pub num_workers: Option<usize>,
}

impl Config {
    /// Parses socket permissions as an octal number. E.g., "666" -> 0o666.
    pub fn socket_permissions(&self) -> anyhow::Result<u32> {
        u32::from_str_radix(&self.socket_permissions, 8)
            .context("Failed to parse socket permissions")
    }
}

/// The mode of connection to use for the daemon.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum ConnectionMode {
    /// Unix domain socket.
    UDS,
    /// TCP.
    TCP,
}

/// Parse command line arguments and return the resulting configuration object.
pub fn get_config() -> Config {
    Config::parse()
}
