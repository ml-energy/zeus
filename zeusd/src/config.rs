//! Configuration.

use clap::Parser;

/// The Zeus daemon runs with elevated provileges and communicates with
/// unprivileged Zeus clients over a Unix domain socket to allow them to
/// interact with and control compute devices on the node.
#[derive(Parser, Debug)]
#[command(version)]
pub struct Config {
    /// Path to the socket Zeusd will listen on.
    #[clap(long, default_value = "/var/run/zeusd.sock")]
    pub socket_path: String,

    /// Unix octal permissions for the socket, e.g. "644", "666".
    /// This will be the permissions of the socket file created by Zeusd.
    #[clap(long, default_value = "666")]
    pub socket_permissions: String,
}

pub fn get_config() -> Config {
    Config::parse()
}
