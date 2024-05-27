//! Entry point for the daemon.

use std::net::TcpListener;

use zeusd::config::{get_config, ConnectionMode};
use zeusd::startup::{
    ensure_root, get_unix_listener, init_tracing, start_device_tasks, start_server_tcp,
    start_server_uds,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing(std::io::stdout)?;

    let config = get_config();
    tracing::info!("Loaded {:?}", config);

    if !config.allow_unprivileged {
        ensure_root()?;
    }

    let device_tasks = start_device_tasks()?;
    tracing::info!("Started device tasks");

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

            start_server_uds(listener, device_tasks, num_workers)?.await?;
        }
        ConnectionMode::TCP => {
            let listener = TcpListener::bind(&config.tcp_bind_address)?;
            tracing::info!("Listening on {}", &config.tcp_bind_address);

            start_server_tcp(listener, device_tasks, num_workers)?.await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// We won't be running tests as root.
    fn test_ensure_root() {
        assert!(ensure_root().is_err());
    }
}
