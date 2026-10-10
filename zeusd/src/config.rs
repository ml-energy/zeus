//! Zeus daemon configuration.

use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

use crate::devices::cpu::power_limit_snapshot::OriginalPowerLimitStorage;

/// API groups that can be independently enabled or disabled.
///
/// Each group maps to a set of HTTP endpoints. Each operation needs its own
/// device and kernel permissions, and fails with an explanation if Zeusd
/// lacks them.
///
/// Available groups:
///   - `gpu-control`: GPU control operations (set power limit, locked clocks,
///     persistence mode).
///     - `POST /gpu/set_persistence_mode`
///     - `POST /gpu/set_power_limit`
///     - `POST /gpu/set_gpu_locked_clocks`
///     - `POST /gpu/reset_gpu_locked_clocks`
///     - `POST /gpu/set_mem_locked_clocks`
///     - `POST /gpu/reset_mem_locked_clocks`
///     - `POST /gpu/reset_locked_clocks`
///   - `gpu-read`: GPU monitoring (power readings, energy consumption, power
///     limits, persistence mode).
///     - `GET /gpu/get_power`
///     - `GET /gpu/stream_power`
///     - `GET /gpu/get_cumulative_energy`
///     - `GET /gpu/get_power_limit`
///     - `GET /gpu/get_power_limit_constraints`
///     - `GET /gpu/get_persistence_mode`
///   - `cpu-read`: CPU RAPL monitoring (energy, power readings, power limits).
///     - `GET /cpu/get_cumulative_energy`
///     - `GET /cpu/get_power`
///     - `GET /cpu/stream_power`
///     - `GET /cpu/get_power_limit`
///     - `GET /cpu/get_power_limit_constraints`
///   - `cpu-control`: CPU power capping with RAPL or AMD HSMP.
///     - `POST /cpu/set_power_limit`
///     - `POST /cpu/set_power_limit_time_window`
///     - `POST /cpu/reset_power_limit`
///
/// The following endpoints are always available regardless of enabled groups:
///   - `GET /discover`
///   - `GET /time`
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ApiGroup {
    /// GPU control operations (set power limit, clocks, persistence mode).
    GpuControl,
    /// GPU read operations (power reading, energy consumption, power limits,
    /// persistence mode).
    GpuRead,
    /// CPU RAPL read operations (energy, power, power limits).
    CpuRead,
    /// CPU power capping (set and reset power limits and time windows).
    CpuControl,
}

impl std::fmt::Display for ApiGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiGroup::GpuControl => write!(f, "gpu-control"),
            ApiGroup::GpuRead => write!(f, "gpu-read"),
            ApiGroup::CpuRead => write!(f, "cpu-read"),
            ApiGroup::CpuControl => write!(f, "cpu-control"),
        }
    }
}

/// GPU management backend selection.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum GpuBackend {
    /// Probe compiled backends and select the one that reports GPUs.
    Auto,
    /// Use NVIDIA Management Library.
    Nvml,
    /// Use AMD SMI.
    Amdsmi,
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackend::Auto => write!(f, "auto"),
            GpuBackend::Nvml => write!(f, "nvml"),
            GpuBackend::Amdsmi => write!(f, "amdsmi"),
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
    /// Operating mode (default depends on platform: `uds` on Unix,
    /// `named-pipe` on Windows).
    #[cfg_attr(unix, clap(long, default_value = "uds"))]
    #[cfg_attr(windows, clap(long, default_value = "named-pipe"))]
    pub mode: ConnectionMode,

    /// [UDS mode] Path to the socket Zeusd will listen on.
    #[cfg(unix)]
    #[clap(long, default_value = "/run/zeusd/zeusd.sock")]
    pub socket_path: String,

    /// [UDS mode] Permissions for the socket file to be created.
    #[cfg(unix)]
    #[clap(long, default_value = "666")]
    socket_permissions: String,

    /// [UDS mode] UID to chown the socket file to.
    #[cfg(unix)]
    #[clap(long)]
    pub socket_uid: Option<u32>,

    /// [UDS mode] GID to chown the socket file to.
    #[cfg(unix)]
    #[clap(long)]
    pub socket_gid: Option<u32>,

    /// [Named pipe mode] Pipe name (Windows only).
    #[cfg(windows)]
    #[clap(long, default_value = r"\\.\pipe\zeusd")]
    pub pipe_name: String,

    /// [Named pipe mode] SDDL for the pipe's DACL. Default grants read+write
    /// to all authenticated users (analogous to `--socket-permissions 666`).
    #[cfg(windows)]
    #[clap(long, default_value = "D:(A;;GRGW;;;AU)")]
    pub pipe_sddl: String,

    /// [TCP mode] Address to bind to.
    #[clap(long, default_value = "127.0.0.1:4938")]
    pub tcp_bind_address: String,

    /// Number of worker threads to use. Default is the number of logical CPUs.
    #[clap(long)]
    pub num_workers: Option<usize>,

    /// GPU power polling frequency in Hz for the streaming endpoint.
    #[clap(long, default_value = "20")]
    pub gpu_power_poll_hz: u32,

    /// GPU backend to use. `auto` probes compiled backends and picks the one
    /// that reports GPUs; an explicit choice fails fast if unavailable.
    #[clap(long, default_value = "auto")]
    pub gpu_backend: GpuBackend,

    /// CPU RAPL power polling frequency in Hz for the streaming endpoint.
    #[clap(long, default_value = "10")]
    pub cpu_power_poll_hz: u32,

    /// API groups to enable. Defaults include the API groups supported by the
    /// platform and compiled device backends.
    #[clap(long, value_delimiter = ',')]
    #[cfg_attr(all(target_os = "linux", any(feature = "nvml", feature = "amdsmi")), clap(
        default_values_t = [ApiGroup::GpuControl, ApiGroup::GpuRead, ApiGroup::CpuRead, ApiGroup::CpuControl],
    ))]
    #[cfg_attr(all(target_os = "linux", not(any(feature = "nvml", feature = "amdsmi"))), clap(
        default_values_t = [ApiGroup::CpuRead, ApiGroup::CpuControl],
    ))]
    #[cfg_attr(all(not(target_os = "linux"), feature = "nvml"), clap(
        default_values_t = [ApiGroup::GpuControl, ApiGroup::GpuRead],
    ))]
    pub enable: Vec<ApiGroup>,

    /// Path to a TOML file mapping GPU control operations to external commands
    /// that are executed instead of the native library call.
    #[clap(long)]
    pub gpu_command_overrides: Option<String>,

    /// Path to the HMAC-SHA256 signing key file for JWT authentication.
    /// If not provided, authentication is disabled.
    #[clap(long)]
    pub signing_key_path: Option<String>,

    /// [cpu-control] Path from which Zeusd names the file of the original CPU
    /// power limit settings that `POST /cpu/reset_power_limit` restores. The
    /// host boot ID is inserted before the extension, e.g.,
    /// `original_cpu_power_limit.<boot ID>.json`. The first Zeusd start in a
    /// boot records the settings it finds in that file as the original
    /// settings, and later starts in the same boot load them, so the directory
    /// should persist across Zeusd restarts and container replacements.
    #[clap(long, default_value = "/var/zeusd/original_cpu_power_limit.json")]
    pub original_cpu_power_limit_path: String,

    /// [cpu-control] Keep the original CPU power limit settings that
    /// `POST /cpu/reset_power_limit` restores in memory instead of in a file,
    /// recording them at every Zeusd start. No writable storage is needed, but
    /// a restarted Zeusd records the settings it finds then as the original
    /// settings, including limits set before the restart.
    #[clap(long, conflicts_with = "original_cpu_power_limit_path")]
    pub no_persistent_original_cpu_power_limit: bool,
}

impl ServeConfig {
    /// Parses socket permissions as an octal number. E.g., "666" -> 0o666.
    #[cfg(unix)]
    pub fn socket_permissions(&self) -> anyhow::Result<u32> {
        u32::from_str_radix(&self.socket_permissions, 8)
            .context("Failed to parse socket permissions")
    }

    /// Whether the given API group is enabled.
    pub fn is_enabled(&self, group: ApiGroup) -> bool {
        self.enable.contains(&group)
    }

    /// Whether any GPU API group is enabled.
    pub fn needs_gpu(&self) -> bool {
        self.is_enabled(ApiGroup::GpuControl) || self.is_enabled(ApiGroup::GpuRead)
    }

    /// Whether any CPU API group is enabled (requiring RAPL initialization).
    pub fn needs_cpu(&self) -> bool {
        self.is_enabled(ApiGroup::CpuRead) || self.is_enabled(ApiGroup::CpuControl)
    }

    /// Where the original CPU power limit settings are kept.
    pub fn original_cpu_power_limit_storage(&self) -> OriginalPowerLimitStorage {
        if self.no_persistent_original_cpu_power_limit {
            OriginalPowerLimitStorage::InMemory
        } else {
            OriginalPowerLimitStorage::Persistent(PathBuf::from(
                &self.original_cpu_power_limit_path,
            ))
        }
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
///
/// Variants are gated by platform: `UDS` is Unix-only and `NamedPipe`
/// is Windows-only. `TCP` is always available.
#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
pub enum ConnectionMode {
    /// Unix domain socket.
    #[cfg(unix)]
    UDS,
    /// TCP.
    TCP,
    /// Windows named pipe (`\\.\pipe\<name>`).
    #[cfg(windows)]
    NamedPipe,
}

/// Parse command line arguments.
pub fn get_cli() -> Cli {
    Cli::parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn serve(args: &[&str]) -> Result<ServeConfig, clap::Error> {
        let cli = Cli::try_parse_from(["zeusd", "serve"].iter().chain(args))?;
        match cli.command {
            Command::Serve(config) => Ok(config),
            Command::Token { .. } => unreachable!(),
        }
    }

    #[test]
    fn original_is_persistent_under_var_by_default() {
        assert_eq!(
            serve(&[]).unwrap().original_cpu_power_limit_storage(),
            OriginalPowerLimitStorage::Persistent(PathBuf::from(
                "/var/zeusd/original_cpu_power_limit.json"
            ))
        );
        assert_eq!(
            serve(&[
                "--original-cpu-power-limit-path",
                "/srv/zeusd/original.json"
            ])
            .unwrap()
            .original_cpu_power_limit_storage(),
            OriginalPowerLimitStorage::Persistent(PathBuf::from("/srv/zeusd/original.json"))
        );
    }

    #[test]
    fn no_persistent_original_flag_keeps_the_original_in_memory() {
        assert_eq!(
            serve(&["--no-persistent-original-cpu-power-limit"])
                .unwrap()
                .original_cpu_power_limit_storage(),
            OriginalPowerLimitStorage::InMemory
        );
        assert!(serve(&[
            "--no-persistent-original-cpu-power-limit",
            "--original-cpu-power-limit-path",
            "/srv/zeusd/original.json",
        ])
        .is_err());
    }
}
