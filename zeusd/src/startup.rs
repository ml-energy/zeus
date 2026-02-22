//! Startup logic.

use actix_web::dev::Server;
use actix_web::{web, App, HttpServer};
use std::collections::HashSet;
use std::fs;
use std::net::TcpListener;
use std::os::unix::fs::{chown, PermissionsExt};
use std::os::unix::net::UnixListener;
use tracing::subscriber::set_global_default;
use tracing_log::LogTracer;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{EnvFilter, Registry};

use crate::auth::{AuthMiddleware, SigningKeyData};
use crate::config::ApiGroup;
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

/// Check that the daemon has sufficient privileges for the requested API groups.
///
/// For each enabled group that requires root, verifies that the effective
/// user ID is 0. Returns an error naming the offending group if not.
pub fn check_privileges(enabled_groups: &[ApiGroup]) -> anyhow::Result<()> {
    let is_root = nix::unistd::geteuid().is_root();
    for &group in enabled_groups {
        if group.requires_root() && !is_root {
            tracing::error!(
                "API group '{}' requires root privileges. \
                 Either run as root or remove it from --enable.",
                group,
            );
            anyhow::bail!(
                "API group '{}' requires root but Zeusd is not running as root",
                group,
            );
        }
    }
    Ok(())
}

/// The set of enabled API groups, used as shared application state.
#[derive(Clone, Debug)]
pub struct EnabledGroups(pub HashSet<ApiGroup>);

/// Shared server state bundling all optional device handles and discovery info.
///
/// Fields are `Option` because devices are only initialized when their
/// corresponding API groups are enabled.
#[derive(Clone)]
pub struct ServerState {
    pub gpu_device_tasks: Option<GpuManagementTasks>,
    pub cpu_device_tasks: Option<CpuManagementTasks>,
    pub gpu_power_broadcast: Option<GpuPowerBroadcast>,
    pub cpu_power_broadcast: Option<CpuPowerBroadcast>,
    pub discovery_info: DiscoveryInfo,
    pub enabled_groups: EnabledGroups,
    pub signing_key: Option<SigningKeyData>,
}

/// Build an `HttpServer` with routes and app data based on enabled API groups.
macro_rules! configure_server {
    ($state:expr, $workers:expr) => {
        HttpServer::new(move || {
            let state = $state.clone();
            let enabled = &state.enabled_groups.0;

            let mut app = App::new()
                .wrap(AuthMiddleware)
                .wrap(tracing_actix_web::TracingLogger::default())
                .configure(server_routes)
                .app_data(web::Data::new(state.discovery_info.clone()))
                .app_data(web::Data::new(state.enabled_groups.clone()));

            // Register signing key for the auth middleware (if configured).
            if let Some(ref key) = state.signing_key {
                app = app.app_data(web::Data::new(key.clone()));
            }

            // GPU routes: conditionally register read and/or control routes.
            if enabled.contains(&ApiGroup::GpuRead) || enabled.contains(&ApiGroup::GpuControl) {
                let mut gpu_scope = web::scope("/gpu");
                if enabled.contains(&ApiGroup::GpuRead) {
                    gpu_scope = gpu_scope.configure(gpu_read_routes);
                }
                if enabled.contains(&ApiGroup::GpuControl) {
                    gpu_scope = gpu_scope.configure(gpu_control_routes);
                }
                app = app.service(gpu_scope);
            }
            if let Some(ref tasks) = state.gpu_device_tasks {
                app = app.app_data(web::Data::new(tasks.clone()));
            }
            if let Some(ref broadcast) = state.gpu_power_broadcast {
                app = app.app_data(web::Data::new(broadcast.clone()));
            }

            // CPU routes: only if cpu-read is enabled.
            if enabled.contains(&ApiGroup::CpuRead) {
                app = app.service(web::scope("/cpu").configure(cpu_routes));
            }
            if let Some(ref tasks) = state.cpu_device_tasks {
                app = app.app_data(web::Data::new(tasks.clone()));
            }
            if let Some(ref broadcast) = state.cpu_power_broadcast {
                app = app.app_data(web::Data::new(broadcast.clone()));
            }

            app
        })
        .workers($workers)
    };
}

/// Set up routing and start the server on a unix domain socket.
pub fn start_server_uds(
    listener: UnixListener,
    state: ServerState,
    num_workers: usize,
) -> std::io::Result<Server> {
    Ok(configure_server!(state, num_workers)
        .listen_uds(listener)?
        .run())
}

/// Set up routing and start the server over TCP.
pub fn start_server_tcp(
    listener: TcpListener,
    state: ServerState,
    num_workers: usize,
) -> std::io::Result<Server> {
    Ok(configure_server!(state, num_workers)
        .listen(listener)?
        .run())
}
