//! Configuration.

use clap::{Parser, ValueEnum};

/// The Zeus daemon runs with elevated provileges and communicates with
/// unprivileged Zeus clients over a Unix domain socket to allow them to
/// interact with and control compute devices on the node.
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
    pub socket_permissions: String,

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

pub fn get_config() -> Config {
    Config::parse()
}

/// The mode of connection to use for the daemon.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum ConnectionMode {
    /// Unix domain socket.
    UDS,
    /// TCP.
    TCP,
}
