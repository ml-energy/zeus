//! Zeus daemon configuration.

use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

/// API groups that can be independently enabled or disabled.
///
/// Each group maps to a set of HTTP endpoints. Groups that require root
/// will cause the daemon to exit at startup if it is not running as root.
///
/// Available groups:
///   - `gpu-control`: GPU control operations (set power limit, locked clocks,
///     persistence mode). Requires root.
///     - `POST /gpu/set_persistence_mode`
///     - `POST /gpu/set_power_limit`
///     - `POST /gpu/set_gpu_locked_clocks`
///     - `POST /gpu/reset_gpu_locked_clocks`
///     - `POST /gpu/set_mem_locked_clocks`
///     - `POST /gpu/reset_mem_locked_clocks`
///   - `gpu-read`: GPU monitoring (power readings, energy consumption).
///     Does not require root.
///     - `GET /gpu/get_power`
///     - `GET /gpu/stream_power`
///     - `GET /gpu/get_cumulative_energy`
///   - `cpu-read`: CPU RAPL monitoring (energy, power readings). Requires root.
///     - `GET /cpu/get_cumulative_energy`
///     - `GET /cpu/get_power`
///     - `GET /cpu/stream_power`
///
/// The following endpoints are always available regardless of enabled groups:
///   - `GET /discover`
///   - `GET /time`
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ApiGroup {
    /// GPU control operations (set power limit, clocks, persistence mode).
    /// Requires root.
    GpuControl,
    /// GPU read operations (power reading, energy consumption).
    GpuRead,
    /// CPU RAPL read operations (energy, power).
    /// Requires root.
    CpuRead,
}

impl ApiGroup {
    /// Whether this API group requires root privileges.
    pub fn requires_root(&self) -> bool {
        matches!(self, ApiGroup::GpuControl | ApiGroup::CpuRead)
    }
}

impl std::fmt::Display for ApiGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiGroup::GpuControl => write!(f, "gpu-control"),
            ApiGroup::GpuRead => write!(f, "gpu-read"),
            ApiGroup::CpuRead => write!(f, "cpu-read"),
        }
    }
}

/// The Zeus daemon manages and monitors compute devices on the node.
#[derive(Parser, Debug)]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level subcommands.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Start the Zeus daemon.
    Serve(ServeConfig),
    /// Token management.
    Token {
        #[command(subcommand)]
        action: TokenCommand,
    },
}

/// Token management subcommands.
#[derive(Subcommand, Debug)]
pub enum TokenCommand {
    /// Issue a new JWT token for a user.
    Issue(TokenIssueConfig),
}

/// Configuration for the `serve` subcommand.
#[derive(Parser, Debug)]
pub struct ServeConfig {
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

    /// Number of worker threads to use. Default is the number of logical CPUs.
    #[clap(long)]
    pub num_workers: Option<usize>,

    /// GPU power polling frequency in Hz for the streaming endpoint.
    #[clap(long, default_value = "20")]
    pub gpu_power_poll_hz: u32,

    /// CPU RAPL power polling frequency in Hz for the streaming endpoint.
    #[clap(long, default_value = "10")]
    pub cpu_power_poll_hz: u32,

    /// API groups to enable. Each group exposes a set of HTTP endpoints.
    /// Groups that require root will cause the daemon to exit at startup
    /// if it is not running as root.
    #[clap(
        long,
        value_delimiter = ',',
        default_values_t = [ApiGroup::GpuControl, ApiGroup::GpuRead, ApiGroup::CpuRead],
    )]
    pub enable: Vec<ApiGroup>,

    /// Path to the HMAC-SHA256 signing key file for JWT authentication.
    /// If not provided, authentication is disabled.
    #[clap(long)]
    pub signing_key_path: Option<String>,
}

impl ServeConfig {
    /// Parses socket permissions as an octal number. E.g., "666" -> 0o666.
    pub fn socket_permissions(&self) -> anyhow::Result<u32> {
        u32::from_str_radix(&self.socket_permissions, 8)
            .context("Failed to parse socket permissions")
    }

    /// Whether the given API group is enabled.
    pub fn is_enabled(&self, group: ApiGroup) -> bool {
        self.enable.contains(&group)
    }

    /// Whether any GPU API group is enabled (requiring NVML initialization).
    pub fn needs_gpu(&self) -> bool {
        self.is_enabled(ApiGroup::GpuControl) || self.is_enabled(ApiGroup::GpuRead)
    }

    /// Whether any CPU API group is enabled (requiring RAPL initialization).
    pub fn needs_cpu(&self) -> bool {
        self.is_enabled(ApiGroup::CpuRead)
    }
}

/// Configuration for the `token issue` subcommand.
#[derive(Parser, Debug)]
pub struct TokenIssueConfig {
    /// Path to the HMAC-SHA256 signing key file.
    #[clap(long)]
    pub signing_key_path: String,

    /// User identity to embed in the token (the `sub` claim).
    #[clap(long)]
    pub user: String,

    /// API group scopes to grant. Comma-separated.
    #[clap(long, value_delimiter = ',')]
    pub scope: Vec<ApiGroup>,

    /// Token lifetime. Human-readable duration (e.g., "1h", "7d", "30d").
    /// Use "never" for tokens that do not expire.
    #[clap(long)]
    pub expires: String,
}

impl TokenIssueConfig {
    /// Parse the `--expires` value into an optional Unix timestamp.
    ///
    /// Returns `None` for "never" or "0" (no expiry), otherwise returns
    /// `Some(unix_timestamp)`.
    pub fn expires_at(&self) -> anyhow::Result<Option<usize>> {
        let s = self.expires.trim().to_lowercase();
        if s == "never" || s == "0" {
            return Ok(None);
        }
        let duration: std::time::Duration = s
            .parse::<humantime::Duration>()
            .context(format!(
                "Invalid duration '{}'. Use e.g. '1h', '7d', '30d', or 'never'.",
                self.expires
            ))?
            .into();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .context("System clock error")?;
        Ok(Some((now + duration).as_secs() as usize))
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

/// Parse command line arguments.
pub fn get_cli() -> Cli {
    Cli::parse()
}
