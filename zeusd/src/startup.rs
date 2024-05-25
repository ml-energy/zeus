//! Startup logic.

use actix_web::dev::Server;
use actix_web::{web, App, HttpServer};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixListener;
use tracing::subscriber::set_global_default;
use tracing_log::LogTracer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{EnvFilter, Registry};

use crate::device::gpu::GpuHandlers;
use crate::routes::gpu::{set_frequency, set_power_limit};
use crate::routes::info::info;

pub fn init_tracing() -> anyhow::Result<()> {
    LogTracer::init()?;

    let subscriber = Registry::default()
        .with(tracing_subscriber::fmt::layer())
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")));
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

/// Start the GPU handlers.
///
/// When we add CPU (RAPL) support, we want to generalize this function to start all handlers.
pub async fn start_device_handlers() -> anyhow::Result<GpuHandlers> {
    GpuHandlers::start().await
}

/// Set up routing and start the server.
pub fn start_server(listener: UnixListener, gpu_handlers: GpuHandlers) -> std::io::Result<Server> {
    let server = HttpServer::new(move || {
        App::new()
            .service(info)
            .service(set_power_limit)
            .service(set_frequency)
            .app_data(web::Data::new(gpu_handlers.clone()))
    })
    .listen_uds(listener)?
    .run();

    Ok(server)
}
