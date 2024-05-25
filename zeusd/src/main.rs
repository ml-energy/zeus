use std::os::unix::net::UnixListener;
use zeusd::config::get_config;
use zeusd::device::gpu::GpuHandlers;
use zeusd::startup::run;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = get_config();

    let listener = UnixListener::bind(&config.socket_path)?;
    let gpu_handlers = GpuHandlers::start();
    run(listener, gpu_handlers).await?;

    Ok(())
}
