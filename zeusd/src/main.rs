//! Entry point for the Zeus daemon.

use std::net::{IpAddr, TcpListener};
use std::sync::Arc;

use zeusd::auth::{issue_token, SigningKeyData};
use zeusd::config::{get_cli, ApiGroup, Command, ConnectionMode, TokenCommand};
use zeusd::devices::gpu::command_override::GpuCommandOverrides;
use zeusd::routes::CpuPowerSamplingPeriod;
use zeusd::routes::DiscoveryInfo;
#[cfg(windows)]
use zeusd::startup::run_server_named_pipe;
use zeusd::startup::{
    check_privileges, init_tracing, resolve_gpu_backend, start_cpu_device_tasks,
    start_cpu_power_poller, start_gpu_device_tasks, start_gpu_power_poller, start_server_tcp,
    EnabledGroups, ServerState,
};
#[cfg(unix)]
use zeusd::startup::{get_unix_listener, start_server_uds};

/// Read the signing key from the given file path.
fn read_signing_key(path: &str) -> anyhow::Result<Vec<u8>> {
    let key = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("Failed to read signing key from '{}': {}", path, e))?;
    if key.is_empty() {
        anyhow::bail!("Signing key file '{}' is empty", path);
    }
    Ok(key)
}

/// Return whether a bound address is loopback, including IPv4-mapped IPv6.
fn is_loopback_ip(ip: IpAddr) -> bool {
    ip.to_canonical().is_loopback()
}

/// Bind a TCP listener and reject remote unauthenticated exposure by default.
fn bind_tcp_listener(
    bind_address: &str,
    authentication_enabled: bool,
    allow_unauthenticated_tcp: bool,
) -> anyhow::Result<TcpListener> {
    let listener = TcpListener::bind(bind_address)?;
    let local_addr = listener.local_addr()?;
    if !authentication_enabled
        && !allow_unauthenticated_tcp
        && !is_loopback_ip(local_addr.ip())
    {
        anyhow::bail!(
            "Refusing unauthenticated TCP listener on non-loopback address '{}'. \
             Configure --signing-key-path or explicitly opt in with \
             --allow-unauthenticated-tcp.",
            local_addr,
        );
    }
    Ok(listener)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing(std::io::stdout)?;

    let cli = get_cli();

    match cli.command {
        Command::Token { action } => handle_token_command(action),
        Command::Serve(config) => handle_serve(config).await,
    }
}

/// Handle `zeusd token issue`.
fn handle_token_command(action: TokenCommand) -> anyhow::Result<()> {
    match action {
        TokenCommand::Issue(config) => {
            let key_bytes = read_signing_key(&config.signing_key_path)?;
            let expires_at = config.expires_at()?;
            let token = issue_token(&key_bytes, &config.user, config.scope.clone(), expires_at)?;
            println!("{token}");
            Ok(())
        }
    }
}

/// Handle `zeusd serve`.
async fn handle_serve(config: zeusd::config::ServeConfig) -> anyhow::Result<()> {
    tracing::info!("Loaded {:?}", config);

    let mut gpu_command_overrides = match &config.gpu_command_overrides {
        Some(path) => Some(Arc::new(GpuCommandOverrides::load(path)?)),
        None => None,
    };
    if gpu_command_overrides.is_some() && !config.is_enabled(ApiGroup::GpuControl) {
        tracing::warn!(
            "GPU command overrides are ignored because the gpu-control API group is not enabled"
        );
        gpu_command_overrides = None;
    }

    let resolved_gpu_backend = if config.needs_gpu() {
        Some(resolve_gpu_backend(config.gpu_backend)?)
    } else {
        None
    };

    // Validate privileges for the requested API groups.
    check_privileges(&config.enable, gpu_command_overrides.as_deref())?;

    let enabled_groups = EnabledGroups(config.enable.iter().cloned().collect());
    tracing::info!(
        "Enabled API groups: {}",
        config
            .enable
            .iter()
            .map(|g| g.to_string())
            .collect::<Vec<_>>()
            .join(", "),
    );

    // Load signing key if provided.
    let signing_key_data = match &config.signing_key_path {
        Some(path) => {
            let key_bytes = read_signing_key(path)?;
            tracing::info!("JWT authentication enabled (key loaded from {})", path);
            Some(SigningKeyData(Arc::new(
                jsonwebtoken::DecodingKey::from_secret(&key_bytes),
            )))
        }
        None => None,
    };
    let auth_required = signing_key_data.is_some();

    // Conditionally initialize GPU devices.
    let (gpu_device_tasks, gpu_power_broadcast, gpus) = if config.needs_gpu() {
        let backend = resolved_gpu_backend.expect("GPU backend must be resolved");
        let (tasks, gpus) = start_gpu_device_tasks(backend, gpu_command_overrides.clone())?;
        let broadcast = if config.is_enabled(ApiGroup::GpuRead) {
            Some(start_gpu_power_poller(backend, config.gpu_power_poll_hz)?)
        } else {
            None
        };
        (Some(tasks), broadcast, gpus)
    } else {
        (None, None, vec![])
    };

    // Conditionally initialize CPU devices.
    let (cpu_device_tasks, cpu_power_broadcast, cpus) = if config.needs_cpu() {
        let (tasks, cpus) = start_cpu_device_tasks()?;
        let broadcast = start_cpu_power_poller(config.cpu_power_poll_hz)?;
        (Some(tasks), Some(broadcast), cpus)
    } else {
        (None, None, vec![])
    };

    tracing::info!("Started all device tasks");

    let discovery_info = DiscoveryInfo {
        gpus,
        cpus,
        enabled_api_groups: config.enable.iter().map(|g| g.to_string()).collect(),
        auth_required,
    };
    tracing::info!("Discovery: {:?}", serde_json::to_string(&discovery_info)?);

    let state = ServerState {
        gpu_device_tasks,
        cpu_device_tasks,
        gpu_power_broadcast,
        cpu_power_broadcast,
        cpu_power_sampling_period: if config.needs_cpu() {
            Some(CpuPowerSamplingPeriod::from_poll_hz(
                config.cpu_power_poll_hz,
            ))
        } else {
            None
        },
        discovery_info,
        enabled_groups,
        signing_key: signing_key_data,
    };

    let num_workers = config.num_workers.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .expect("Failed to get number of logical CPUs")
            .into()
    });
    match config.mode {
        #[cfg(unix)]
        ConnectionMode::UDS => {
            let listener = get_unix_listener(
                &config.socket_path,
                config.socket_permissions()?,
                config.socket_uid,
                config.socket_gid,
            )?;
            tracing::info!("Listening on {}", &config.socket_path);

            start_server_uds(listener, state, num_workers)?.await?;
        }
        ConnectionMode::TCP => {
            let listener = bind_tcp_listener(
                &config.tcp_bind_address,
                auth_required,
                config.allow_unauthenticated_tcp,
            )?;
            let local_addr = listener.local_addr()?;
            if !auth_required {
                if config.allow_unauthenticated_tcp {
                    tracing::warn!(
                        "Running unauthenticated TCP on {} because \
                         --allow-unauthenticated-tcp was explicitly set.",
                        local_addr,
                    );
                } else {
                    tracing::info!("Running unauthenticated loopback TCP on {}", local_addr);
                }
            }
            tracing::info!("Listening on {}", local_addr);

            start_server_tcp(listener, state, num_workers)?.await?;
        }
        #[cfg(windows)]
        ConnectionMode::NamedPipe => {
            let _ = num_workers; // num_workers applies to actix-server workers (TCP/UDS)
            run_server_named_pipe(config.pipe_name.clone(), config.pipe_sddl.clone(), state)
                .await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_detection_canonicalizes_ipv4_mapped_ipv6() {
        assert!(is_loopback_ip("127.0.0.1".parse().unwrap()));
        assert!(is_loopback_ip("::1".parse().unwrap()));
        assert!(is_loopback_ip("::ffff:127.0.0.1".parse().unwrap()));
        assert!(!is_loopback_ip("0.0.0.0".parse().unwrap()));
        assert!(!is_loopback_ip("::".parse().unwrap()));
    }

    #[test]
    fn unauthenticated_tcp_allows_loopback() {
        let listener = bind_tcp_listener("127.0.0.1:0", false, false).unwrap();
        assert!(is_loopback_ip(listener.local_addr().unwrap().ip()));
    }

    #[test]
    fn unauthenticated_tcp_rejects_non_loopback() {
        let error = bind_tcp_listener("0.0.0.0:0", false, false).unwrap_err();
        assert!(error.to_string().contains("Refusing unauthenticated TCP"));
        assert!(error.to_string().contains("--signing-key-path"));
    }

    #[test]
    fn authentication_allows_non_loopback() {
        bind_tcp_listener("0.0.0.0:0", true, false).unwrap();
    }

    #[test]
    fn explicit_opt_in_allows_unauthenticated_non_loopback() {
        bind_tcp_listener("0.0.0.0:0", false, true).unwrap();
    }
}
