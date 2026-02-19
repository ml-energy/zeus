//! GPU power polling and streaming.
//!
//! A dedicated background task polls GPU power at a configurable frequency
//! and broadcasts snapshots via a `tokio::sync::watch` channel. Polling is
//! demand-driven: the task sleeps when no subscribers are connected and wakes
//! when the first subscriber arrives.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tokio::sync::{watch, Notify};
use tokio::time::{interval, Duration};

use crate::devices::gpu::GpuManager;
use crate::power_streaming::{PowerBroadcast, PowerPoller};

/// A snapshot of GPU power readings across all monitored GPUs.
#[derive(Clone, Debug, Default, Serialize)]
pub struct GpuPowerSnapshot {
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Power readings in milliwatts, keyed by GPU index.
    pub power_mw: BTreeMap<usize, u32>,
}

/// Broadcast handle for GPU power snapshots.
pub type GpuPowerBroadcast = PowerBroadcast<GpuPowerSnapshot>;

/// Background poller for GPU power.
pub type GpuPowerPoller = PowerPoller<GpuPowerSnapshot>;

/// Start the GPU power polling background task.
///
/// Creates a dedicated tokio task that reads power from each GPU at
/// `poll_hz` frequency when subscribers are present. The task sleeps
/// when no subscribers are connected.
pub fn start_gpu_poller<T: GpuManager + Send + 'static>(
    gpus: Vec<(usize, T)>,
    poll_hz: u32,
) -> GpuPowerPoller {
    let valid_ids = gpus.iter().map(|(idx, _)| *idx).collect();
    PowerPoller::start(valid_ids, |tx, subscriber_count, wake| {
        gpu_power_poll_task(gpus, tx, poll_hz, subscriber_count, wake)
    })
}

async fn gpu_power_poll_task<T: GpuManager>(
    mut gpus: Vec<(usize, T)>,
    tx: watch::Sender<GpuPowerSnapshot>,
    poll_hz: u32,
    subscriber_count: Arc<AtomicUsize>,
    wake: Arc<Notify>,
) {
    if gpus.is_empty() {
        tracing::info!("No GPUs to monitor, power poller idle");
        // Hold tx alive so subscribers don't see RecvError.
        std::future::pending::<()>().await;
        return;
    }

    let period_us = 1_000_000u64 / poll_hz.max(1) as u64;
    let mut last_power: BTreeMap<usize, u32> = BTreeMap::new();

    tracing::info!(
        "GPU power poller ready: {} GPUs at {} Hz when subscribers are present",
        gpus.len(),
        poll_hz
    );

    loop {
        // Sleep until at least one subscriber connects.
        while subscriber_count.load(Ordering::Relaxed) == 0 {
            wake.notified().await;
        }

        tracing::info!("GPU power poller starting");
        let mut tick = interval(Duration::from_micros(period_us));

        // Poll while subscribers are present.
        while subscriber_count.load(Ordering::Relaxed) > 0 {
            tick.tick().await;
            let mut current_power = BTreeMap::new();
            let mut changed = false;

            for (idx, gpu) in gpus.iter_mut() {
                match gpu.get_instant_power_mw() {
                    Ok(power_mw) => {
                        if last_power.get(idx) != Some(&power_mw) {
                            changed = true;
                        }
                        current_power.insert(*idx, power_mw);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to read power for GPU {}: {}", idx, e);
                        if let Some(&last) = last_power.get(idx) {
                            current_power.insert(*idx, last);
                        }
                    }
                }
            }

            // Send on first poll (last_power empty) or on any change.
            if changed || last_power.is_empty() {
                let timestamp_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                last_power.clone_from(&current_power);
                let _ = tx.send(GpuPowerSnapshot {
                    timestamp_ms,
                    power_mw: current_power,
                });
            }
        }

        last_power.clear();
        tracing::info!("GPU power poller pausing (no subscribers)");
    }
}
