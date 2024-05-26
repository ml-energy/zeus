//! Startup logic.

use actix_web::dev::Server;
use actix_web::{web, App, HttpServer};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixListener;
use tracing::subscriber::set_global_default;
use tracing_log::LogTracer;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{EnvFilter, Registry};

use crate::devices::gpu::{GpuManagementTasks, GpuManager, NvmlGpu};
use crate::routes::gpu_routes;

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
pub fn get_listener(socket_path: &str) -> anyhow::Result<UnixListener> {
    if fs::metadata(socket_path).is_ok() {
        anyhow::bail!(
            "Socket file {} already exists. Please remove it and restart Zeusd.",
            socket_path,
        );
    }
    let listener = UnixListener::bind(socket_path)?;
    fs::set_permissions(socket_path, fs::Permissions::from_mode(0o666))?;
    Ok(listener)
}

/// Initialize NVML and start GPU management tasks.
///
/// When we add CPU (RAPL) support, we want to generalize this function to start all handlers.
pub fn start_device_handlers() -> anyhow::Result<GpuManagementTasks> {
    let num_gpus = NvmlGpu::device_count()?;
    let mut gpus = Vec::with_capacity(num_gpus as usize);
    for gpu_id in 0..num_gpus {
        let gpu = NvmlGpu::init(gpu_id)?;
        tracing::info!("Initialized NVML for GPU {}", gpu_id);
        gpus.push(gpu);
    }
    GpuManagementTasks::start(gpus)
}

/// Set up routing and start the server.
pub fn start_server(
    listener: UnixListener,
    gpu_handlers: GpuManagementTasks,
) -> std::io::Result<Server> {
    let server = HttpServer::new(move || {
        App::new()
            .wrap(tracing_actix_web::TracingLogger::default())
            .service(web::scope("/gpu").configure(gpu_routes))
            .app_data(web::Data::new(gpu_handlers.clone()))
    })
    .listen_uds(listener)?
    .run();

    Ok(server)
}
