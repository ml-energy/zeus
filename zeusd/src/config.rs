//! Configuration for the Zeus daemon.

use clap::Parser;

/// Configuration struct parsed from command line arguments.
#[derive(Parser, Debug)]
#[command(version)]
pub struct Config {
    #[clap(short, long, default_value = "/var/run/zeusd.sock")]
    pub socket_path: String,
}

pub fn get_config() -> Config {
    Config::parse()
}
