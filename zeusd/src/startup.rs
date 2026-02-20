//! Startup logic.

use actix_web::dev::Server;
use actix_web::{web, App, HttpServer};
use std::fs;
use std::net::TcpListener;
use std::os::unix::fs::{chown, PermissionsExt};
use std::os::unix::net::UnixListener;
use tracing::subscriber::set_global_default;
use tracing_log::LogTracer;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{EnvFilter, Registry};

use crate::devices::cpu::power::{start_cpu_poller, CpuPowerBroadcast, CpuPowerPoller};
use crate::devices::cpu::{CpuManagementTasks, CpuManager, RaplCpu};
use crate::devices::gpu::power::{start_gpu_poller, GpuPowerBroadcast, GpuPowerPoller};
use crate::devices::gpu::{GpuManagementTasks, GpuManager, NvmlGpu};
use crate::routes::cpu_routes;
use crate::routes::{gpu_control_routes, gpu_read_routes, server_routes, DiscoveryInfo};

/// Initialize tracing with the given where to write logs to.
pub fn init_tracing<S>(sink: S) -> anyhow::Result<()>
where
    S: for<'a> MakeWriter<'a> + Send + Sync + 'static,
{
    LogTracer::init()?;

    let formatter = tracing_subscriber::fmt::layer().with_writer(sink);
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let subscriber = Registry::default().with(formatter).with(env_filter);
    set_global_default(subscriber)?;

    Ok(())
}

/// Create a socket at the given path and bind a UnixListener to it.
pub fn get_unix_listener(
    socket_path: &str,
    permissions: u32,
    uid: Option<u32>,
    gid: Option<u32>,
) -> anyhow::Result<UnixListener> {
    if fs::metadata(socket_path).is_ok() {
        tracing::error!(
            "Socket file {} already exists. Please remove it and restart Zeusd.",
            socket_path,
        );
        anyhow::bail!("Socket file already exists");
    }
    let listener = UnixListener::bind(socket_path)?;
    fs::set_permissions(socket_path, fs::Permissions::from_mode(permissions))?;
    chown(socket_path, uid, gid)?;
    Ok(listener)
}

/// Initialize NVML and start GPU management tasks.
pub fn start_gpu_device_tasks() -> anyhow::Result<GpuManagementTasks> {
    tracing::info!("Starting NVML and GPU management tasks.");
    let num_gpus = NvmlGpu::device_count()?;
    let mut gpus = Vec::with_capacity(num_gpus as usize);
    for gpu_id in 0..num_gpus {
        let gpu = NvmlGpu::init(gpu_id)?;
        tracing::info!("Initialized NVML for GPU {}", gpu_id);
        gpus.push(gpu);
    }
    Ok(GpuManagementTasks::start(gpus)?)
}

/// Initialize a separate set of NVML handles and start the GPU power poller.
pub fn start_gpu_power_poller(poll_hz: u32) -> anyhow::Result<GpuPowerPoller> {
    tracing::info!("Starting GPU power poller at {} Hz.", poll_hz);
    let num_gpus = NvmlGpu::device_count()?;
    let mut gpus = Vec::with_capacity(num_gpus as usize);
    for gpu_id in 0..num_gpus {
        let gpu = NvmlGpu::init(gpu_id)?;
        gpus.push((gpu_id as usize, gpu));
    }
    Ok(start_gpu_poller(gpus, poll_hz))
}

/// Initialize RAPL and start CPU management tasks.
///
/// Returns the management tasks and a per-CPU DRAM availability vector.
pub fn start_cpu_device_tasks() -> anyhow::Result<(CpuManagementTasks, Vec<bool>)> {
    tracing::info!("Starting Rapl and CPU management tasks.");
    let num_cpus = RaplCpu::device_count()?;
    let mut cpus = Vec::with_capacity(num_cpus);
    let mut dram_available = Vec::with_capacity(num_cpus);
    for cpu_id in 0..num_cpus {
        let cpu = RaplCpu::init(cpu_id)?;
        dram_available.push(cpu.is_dram_available());
        tracing::info!(
            "Initialized RAPL for CPU {} (DRAM: {})",
            cpu_id,
            dram_available[cpu_id],
        );
        cpus.push(cpu);
    }
    Ok((CpuManagementTasks::start(cpus)?, dram_available))
}

/// Initialize a separate set of RAPL handles and start the CPU power poller.
pub fn start_cpu_power_poller(poll_hz: u32) -> anyhow::Result<CpuPowerPoller> {
    tracing::info!("Starting CPU RAPL power poller at {} Hz.", poll_hz);
    let num_cpus = RaplCpu::device_count()?;
    let mut cpus = Vec::with_capacity(num_cpus);
    for cpu_id in 0..num_cpus {
        let cpu = RaplCpu::init(cpu_id)?;
        cpus.push((cpu_id, cpu));
    }
    Ok(start_cpu_poller(cpus, poll_hz))
}

/// Ensure the daemon is running as root.
pub fn ensure_root() -> anyhow::Result<()> {
    if !nix::unistd::geteuid().is_root() {
        tracing::error!(
            "Zeusd must be run as root to be able to change GPU settings. \
            If you're sure you want to run as non-root, use --allow-unprivileged."
        );
        anyhow::bail!("Zeusd must be run as root");
    }
    Ok(())
}

/// Build an `HttpServer` with all routes and shared application state.
macro_rules! configure_server {
    ($gpu_tasks:expr, $cpu_tasks:expr, $gpu_power:expr, $cpu_power:expr, $discovery:expr, $workers:expr, $monitor_only:expr) => {
        HttpServer::new(move || {
            let monitor_only: bool = $monitor_only;
            let gpu_scope = if monitor_only {
                web::scope("/gpu").configure(gpu_read_routes)
            } else {
                web::scope("/gpu")
                    .configure(gpu_read_routes)
                    .configure(gpu_control_routes)
            };
            App::new()
                .wrap(tracing_actix_web::TracingLogger::default())
                .configure(server_routes)
                .service(gpu_scope)
                .service(web::scope("/cpu").configure(cpu_routes))
                .app_data(web::Data::new($gpu_tasks.clone()))
                .app_data(web::Data::new($cpu_tasks.clone()))
                .app_data(web::Data::new($gpu_power.clone()))
                .app_data(web::Data::new($cpu_power.clone()))
                .app_data(web::Data::new($discovery.clone()))
        })
        .workers($workers)
    };
}

/// Set up routing and start the server on a unix domain socket.
#[allow(clippy::too_many_arguments)]
pub fn start_server_uds(
    listener: UnixListener,
    gpu_device_tasks: GpuManagementTasks,
    cpu_device_tasks: CpuManagementTasks,
    gpu_power_broadcast: GpuPowerBroadcast,
    cpu_power_broadcast: CpuPowerBroadcast,
    discovery_info: DiscoveryInfo,
    num_workers: usize,
    monitor_only: bool,
) -> std::io::Result<Server> {
    Ok(configure_server!(
        gpu_device_tasks,
        cpu_device_tasks,
        gpu_power_broadcast,
        cpu_power_broadcast,
        discovery_info,
        num_workers,
        monitor_only
    )
    .listen_uds(listener)?
    .run())
}

/// Set up routing and start the server over TCP.
#[allow(clippy::too_many_arguments)]
pub fn start_server_tcp(
    listener: TcpListener,
    gpu_device_tasks: GpuManagementTasks,
    cpu_device_tasks: CpuManagementTasks,
    gpu_power_broadcast: GpuPowerBroadcast,
    cpu_power_broadcast: CpuPowerBroadcast,
    discovery_info: DiscoveryInfo,
    num_workers: usize,
    monitor_only: bool,
) -> std::io::Result<Server> {
    Ok(configure_server!(
        gpu_device_tasks,
        cpu_device_tasks,
        gpu_power_broadcast,
        cpu_power_broadcast,
        discovery_info,
        num_workers,
        monitor_only
    )
    .listen(listener)?
    .run())
}
