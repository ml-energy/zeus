//! Entry point for the Zeus daemon.

use std::net::TcpListener;

use zeusd::config::{get_config, ApiGroup, ConnectionMode};
use zeusd::routes::DiscoveryInfo;
use zeusd::startup::{
    check_privileges, get_unix_listener, init_tracing, start_cpu_device_tasks,
    start_cpu_power_poller, start_gpu_device_tasks, start_gpu_power_poller, start_server_tcp,
    start_server_uds, EnabledGroups, ServerState,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing(std::io::stdout)?;

    let config = get_config();
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
    };
    tracing::info!("Discovery: {:?}", serde_json::to_string(&discovery_info)?);

    let state = ServerState {
        gpu_device_tasks,
        cpu_device_tasks,
        gpu_power_broadcast,
        cpu_power_broadcast,
        discovery_info,
        enabled_groups,
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
