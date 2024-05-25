//! Entry point for the daemon.

use zeusd::config::get_config;
use zeusd::startup::{get_listener, init_tracing, start_device_handlers, start_server};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing()?;

    let config = get_config();

    let listener = get_listener(&config.socket_path)?;
    let device_handlers = start_device_handlers().await?;

    tracing::info!("Starting Zeusd...");
    start_server(listener, device_handlers)?.await?;

    Ok(())
}
