//! GPU power polling and streaming.
//!
//! A dedicated background task polls GPU power at a configurable frequency
//! and broadcasts snapshots via a `tokio::sync::watch` channel. Polling is
//! demand-driven: the task sleeps when no subscribers are connected and wakes
//! when the first subscriber arrives.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use serde::Serialize;
use tokio::sync::{watch, Notify};
use tokio::time::{interval, Duration};

use crate::devices::gpu::GpuManager;
use crate::power_streaming::{unix_timestamp_ms, PowerBroadcast, PowerBroadcasts, PowerPoller};

/// A snapshot of GPU power readings across all monitored GPUs.
#[derive(Clone, Debug, Default, Serialize)]
pub struct GpuPowerSnapshot {
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Power readings in milliwatts, keyed by GPU index.
    pub power_mw: BTreeMap<usize, u32>,
}

/// A single GPU power sample for SSE streaming.
#[derive(Clone, Debug, Default, Serialize)]
pub struct GpuPowerSample {
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// GPU index.
    pub gpu_id: usize,
    /// Power reading in milliwatts.
    pub power_mw: u32,
}

/// Broadcast handle for GPU power samples.
pub type GpuPowerBroadcast = PowerBroadcast<GpuPowerSample>;

/// GPU power broadcasts keyed by GPU index.
pub type GpuPowerBroadcasts = PowerBroadcasts<GpuPowerSample>;

/// Background poller for one GPU's power.
pub type GpuPowerPoller = PowerPoller<GpuPowerSample>;

/// Start the GPU power polling background task.
///
/// Creates a dedicated tokio task that reads power from each GPU at
/// `poll_hz` frequency when subscribers are present. The task sleeps
/// when no subscribers are connected.
pub fn start_gpu_poller<T: GpuManager + Send + 'static>(
    gpus: Vec<(usize, T)>,
    poll_hz: u32,
) -> GpuPowerBroadcasts {
    let mut broadcasts = BTreeMap::new();
    for (gpu_id, gpu) in gpus {
        let poller = PowerPoller::start(move |tx, subscriber_count, wake| {
            gpu_power_poll_task(gpu_id, gpu, tx, poll_hz, subscriber_count, wake)
        });
        broadcasts.insert(gpu_id, poller.broadcast());
    }
    PowerBroadcasts::new(broadcasts)
}

async fn gpu_power_poll_task<T: GpuManager>(
    gpu_id: usize,
    mut gpu: T,
    tx: watch::Sender<GpuPowerSample>,
    poll_hz: u32,
    subscriber_count: Arc<AtomicUsize>,
    wake: Arc<Notify>,
) {
    let period_us = 1_000_000u64 / poll_hz.max(1) as u64;
    let mut last_power: Option<u32> = None;

    tracing::info!(
        "GPU power poller ready for GPU {} at {} Hz when subscribers are present",
        gpu_id,
        poll_hz
    );

    loop {
        // Sleep until at least one subscriber connects.
        while subscriber_count.load(Ordering::Relaxed) == 0 {
            wake.notified().await;
        }

        tracing::info!("GPU power poller starting for GPU {}", gpu_id);
        let mut tick = interval(Duration::from_micros(period_us));

        // Poll while subscribers are present.
        while subscriber_count.load(Ordering::Relaxed) > 0 {
            tick.tick().await;
            match gpu.get_instant_power_mw() {
                Ok(power_mw) => {
                    if last_power == Some(power_mw) {
                        continue;
                    }
                    last_power = Some(power_mw);
                    let _ = tx.send(GpuPowerSample {
                        timestamp_ms: unix_timestamp_ms(),
                        gpu_id,
                        power_mw,
                    });
                }
                Err(e) => {
                    tracing::warn!("Failed to read power for GPU {}: {}", gpu_id, e);
                }
            }
        }

        last_power = None;
        tracing::info!("GPU power poller pausing for GPU {}", gpu_id);
    }
}
