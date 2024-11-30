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

use crate::devices::cpu::{CpuManagementTasks, CpuManager, RaplCpu};
use crate::devices::gpu::{GpuManagementTasks, GpuManager, NvmlGpu};
use crate::routes::cpu_routes;
use crate::routes::gpu_routes;

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

pub fn start_cpu_device_tasks() -> anyhow::Result<CpuManagementTasks> {
    tracing::info!("Starting Rapl and CPU management tasks.");
    let num_cpus = RaplCpu::device_count()?;
    let mut cpus = Vec::with_capacity(num_cpus);
    for cpu_id in 0..num_cpus {
        let cpu = RaplCpu::init(cpu_id)?;
        tracing::info!("Initialized RAPL for CPU {}", cpu_id);
        cpus.push(cpu);
    }
    Ok(CpuManagementTasks::start(cpus)?)
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

/// Set up routing and start the server on a unix domain socket.
pub fn start_server_uds(
    listener: UnixListener,
    gpu_device_tasks: GpuManagementTasks,
    cpu_device_tasks: CpuManagementTasks,
    num_workers: usize,
) -> std::io::Result<Server> {
    let server = HttpServer::new(move || {
        App::new()
            .wrap(tracing_actix_web::TracingLogger::default())
            .service(web::scope("/gpu").configure(gpu_routes))
            .service(web::scope("/cpu").configure(cpu_routes))
            .app_data(web::Data::new(gpu_device_tasks.clone()))
            .app_data(web::Data::new(cpu_device_tasks.clone()))
    })
    .workers(num_workers)
    .listen_uds(listener)?
    .run();

    Ok(server)
}

/// Set up routing and start the server over TCP.
pub fn start_server_tcp(
    listener: TcpListener,
    gpu_device_tasks: GpuManagementTasks,
    cpu_device_tasks: CpuManagementTasks,
    num_workers: usize,
) -> std::io::Result<Server> {
    let server = HttpServer::new(move || {
        App::new()
            .wrap(tracing_actix_web::TracingLogger::default())
            .service(web::scope("/gpu").configure(gpu_routes))
            .service(web::scope("/cpu").configure(cpu_routes))
            .app_data(web::Data::new(gpu_device_tasks.clone()))
            .app_data(web::Data::new(cpu_device_tasks.clone()))
    })
    .workers(num_workers)
    .listen(listener)?
    .run();

    Ok(server)
}
