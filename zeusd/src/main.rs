//! Entry point for the Zeus daemon.

use std::net::TcpListener;
use std::sync::Arc;

use zeusd::auth::{issue_token, SigningKeyData};
use zeusd::config::{get_cli, ApiGroup, Command, ConnectionMode, TokenCommand};
use zeusd::routes::DiscoveryInfo;
use zeusd::startup::{
    check_privileges, get_unix_listener, init_tracing, start_cpu_device_tasks,
    start_cpu_power_poller, start_gpu_device_tasks, start_gpu_power_poller, start_server_tcp,
    start_server_uds, EnabledGroups, ServerState,
};

/// Read the signing key from the given file path.
fn read_signing_key(path: &str) -> anyhow::Result<Vec<u8>> {
    let key = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("Failed to read signing key from '{}': {}", path, e))?;
    if key.is_empty() {
        anyhow::bail!("Signing key file '{}' is empty", path);
    }
    Ok(key)
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

    // Validate privileges for the requested API groups.
    check_privileges(&config.enable)?;

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
        None => {
            if config.mode == ConnectionMode::TCP {
                tracing::warn!(
                    "Running in TCP mode without authentication. \
                     Set --signing-key-path for production use."
                );
            }
            None
        }
    };

    // Conditionally initialize GPU devices.
    let (gpu_device_tasks, gpu_power_broadcast, gpu_count) = if config.needs_gpu() {
        let tasks = start_gpu_device_tasks()?;
        let count = tasks.device_count();
        let broadcast = if config.is_enabled(ApiGroup::GpuRead) {
            let poller = start_gpu_power_poller(config.gpu_power_poll_hz)?;
            Some(poller.broadcast())
        } else {
            None
        };
        (Some(tasks), broadcast, count)
    } else {
        (None, None, 0)
    };

    // Conditionally initialize CPU devices.
    let (cpu_device_tasks, cpu_power_broadcast, cpu_count, dram_available) = if config.needs_cpu() {
        let (tasks, dram) = start_cpu_device_tasks()?;
        let count = tasks.device_count();
        let poller = start_cpu_power_poller(config.cpu_power_poll_hz)?;
        let broadcast = poller.broadcast();
        (Some(tasks), Some(broadcast), count, dram)
    } else {
        (None, None, 0, vec![])
    };

    tracing::info!("Started all device tasks");

    let discovery_info = DiscoveryInfo {
        gpu_ids: (0..gpu_count).collect(),
        cpu_ids: (0..cpu_count).collect(),
        dram_available,
        enabled_api_groups: config.enable.iter().map(|g| g.to_string()).collect(),
        auth_required: signing_key_data.is_some(),
    };
    tracing::info!("Discovery: {:?}", serde_json::to_string(&discovery_info)?);

    let state = ServerState {
        gpu_device_tasks,
        cpu_device_tasks,
        gpu_power_broadcast,
        cpu_power_broadcast,
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
            let listener = TcpListener::bind(&config.tcp_bind_address)?;
            tracing::info!("Listening on {}", &listener.local_addr()?);

            start_server_tcp(listener, state, num_workers)?.await?;
        }
    }

    Ok(())
}
