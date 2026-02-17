//! CPU RAPL power polling and streaming.
//!
//! A dedicated background task polls CPU package and DRAM energy counters
//! at a configurable frequency, computes instantaneous power from energy
//! deltas, and broadcasts snapshots via a `tokio::sync::watch` channel.
//! Polling is demand-driven: the task sleeps when no subscribers are
//! connected and wakes when the first subscriber arrives.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tokio::sync::{watch, Notify};
use tokio::task::JoinHandle;
use tokio::time::{interval, Duration};

use crate::devices::cpu::CpuManager;

/// Per-CPU power reading (package + optional DRAM).
#[derive(Clone, Debug, Serialize)]
pub struct CpuDramPower {
    /// CPU package power in milliwatts.
    pub cpu_mw: u32,
    /// DRAM power in milliwatts, if available.
    pub dram_mw: Option<u32>,
}

/// A snapshot of CPU power readings across all monitored CPUs.
#[derive(Clone, Debug, Default, Serialize)]
pub struct CpuPowerSnapshot {
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Power readings keyed by CPU index.
    pub power_mw: HashMap<usize, CpuDramPower>,
}

/// RAII guard that decrements the subscriber count on drop.
pub struct CpuSubscriberGuard {
    count: Arc<AtomicUsize>,
}

impl Drop for CpuSubscriberGuard {
    fn drop(&mut self) {
        self.count.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Shared handle for subscribing to CPU power snapshots.
#[derive(Clone)]
pub struct CpuPowerBroadcast {
    rx: watch::Receiver<CpuPowerSnapshot>,
    subscriber_count: Arc<AtomicUsize>,
    wake: Arc<Notify>,
}

impl CpuPowerBroadcast {
    /// Get the latest power snapshot without waiting for a change.
    pub fn latest(&self) -> CpuPowerSnapshot {
        self.rx.borrow().clone()
    }

    /// Create a new watch receiver for streaming.
    pub fn subscribe(&self) -> watch::Receiver<CpuPowerSnapshot> {
        self.rx.clone()
    }

    /// Register a subscriber, waking the poller if it was sleeping.
    ///
    /// Returns an RAII guard that decrements the count on drop.
    pub fn add_subscriber(&self) -> CpuSubscriberGuard {
        self.subscriber_count.fetch_add(1, Ordering::Relaxed);
        self.wake.notify_one();
        CpuSubscriberGuard {
            count: self.subscriber_count.clone(),
        }
    }
}

/// Background task that polls CPU RAPL counters at a configured frequency,
/// computes instantaneous power, and broadcasts via a watch channel.
pub struct CpuPowerPoller {
    broadcast: CpuPowerBroadcast,
    _handle: JoinHandle<()>,
}

impl CpuPowerPoller {
    /// Start the CPU power polling background task.
    ///
    /// Each CPU must implement `CpuManager`. The poller reads cumulative
    /// energy counters and computes instantaneous power from deltas.
    /// Polling is demand-driven: sleeps when no subscribers are connected.
    pub fn start<T: CpuManager + Send + 'static>(cpus: Vec<(usize, T)>, poll_hz: u32) -> Self {
        let (tx, rx) = watch::channel(CpuPowerSnapshot::default());
        let subscriber_count = Arc::new(AtomicUsize::new(0));
        let wake = Arc::new(Notify::new());
        let handle = tokio::spawn(cpu_power_poll_task(
            cpus,
            tx,
            poll_hz,
            subscriber_count.clone(),
            wake.clone(),
        ));
        Self {
            broadcast: CpuPowerBroadcast {
                rx,
                subscriber_count,
                wake,
            },
            _handle: handle,
        }
    }

    /// Get the broadcast handle for sharing with route handlers.
    pub fn broadcast(&self) -> CpuPowerBroadcast {
        self.broadcast.clone()
    }
}

/// Per-CPU energy tracking state for computing power from energy deltas.
struct CpuEnergyState {
    last_cpu_energy_uj: Option<u64>,
    last_dram_energy_uj: Option<u64>,
    last_cpu_power_mw: u32,
    last_dram_power_mw: Option<u32>,
}

async fn cpu_power_poll_task<T: CpuManager>(
    mut cpus: Vec<(usize, T)>,
    tx: watch::Sender<CpuPowerSnapshot>,
    poll_hz: u32,
    subscriber_count: Arc<AtomicUsize>,
    wake: Arc<Notify>,
) {
    if cpus.is_empty() {
        tracing::info!("No CPUs to monitor, RAPL power poller idle");
        std::future::pending::<()>().await;
        return;
    }

    let period_us = 1_000_000u64 / poll_hz.max(1) as u64;

    // Initialize energy tracking state.
    let mut states: HashMap<usize, CpuEnergyState> = HashMap::new();
    for (idx, _) in &cpus {
        states.insert(
            *idx,
            CpuEnergyState {
                last_cpu_energy_uj: None,
                last_dram_energy_uj: None,
                last_cpu_power_mw: 0,
                last_dram_power_mw: None,
            },
        );
    }

    tracing::info!(
        "CPU RAPL power poller ready: {} CPUs at {} Hz (demand-driven)",
        cpus.len(),
        poll_hz
    );

    loop {
        // Sleep until at least one subscriber connects.
        while subscriber_count.load(Ordering::Relaxed) == 0 {
            wake.notified().await;
        }

        tracing::debug!("CPU power poller waking up");
        let mut tick = interval(Duration::from_micros(period_us));

        // Poll while subscribers are present.
        while subscriber_count.load(Ordering::Relaxed) > 0 {
            tick.tick().await;
            let mut current_power = HashMap::with_capacity(cpus.len());
            let mut changed = false;

            for (idx, cpu) in cpus.iter_mut() {
                let state = states.get_mut(idx).unwrap();

                // Read CPU package energy.
                let cpu_power_mw = match cpu.get_cpu_energy() {
                    Ok(energy_uj) => {
                        if let Some(last_energy) = state.last_cpu_energy_uj {
                            let delta_uj = energy_uj.saturating_sub(last_energy);
                            let power_uw = delta_uj as f64 / (period_us as f64 / 1_000_000.0);
                            let power_mw = (power_uw / 1000.0) as u32;
                            if power_mw != state.last_cpu_power_mw {
                                changed = true;
                            }
                            state.last_cpu_power_mw = power_mw;
                            state.last_cpu_energy_uj = Some(energy_uj);
                            power_mw
                        } else {
                            state.last_cpu_energy_uj = Some(energy_uj);
                            changed = true;
                            0
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to read CPU {} energy: {}", idx, e);
                        state.last_cpu_power_mw
                    }
                };

                // Read DRAM energy if available.
                let dram_power_mw = if cpu.is_dram_available() {
                    match cpu.get_dram_energy() {
                        Ok(energy_uj) => {
                            if let Some(last_energy) = state.last_dram_energy_uj {
                                let delta_uj = energy_uj.saturating_sub(last_energy);
                                let power_uw = delta_uj as f64 / (period_us as f64 / 1_000_000.0);
                                let power_mw = (power_uw / 1000.0) as u32;
                                if state.last_dram_power_mw != Some(power_mw) {
                                    changed = true;
                                }
                                state.last_dram_power_mw = Some(power_mw);
                                state.last_dram_energy_uj = Some(energy_uj);
                                Some(power_mw)
                            } else {
                                state.last_dram_energy_uj = Some(energy_uj);
                                changed = true;
                                Some(0)
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Failed to read CPU {} DRAM energy: {}", idx, e);
                            state.last_dram_power_mw
                        }
                    }
                } else {
                    None
                };

                current_power.insert(
                    *idx,
                    CpuDramPower {
                        cpu_mw: cpu_power_mw,
                        dram_mw: dram_power_mw,
                    },
                );
            }

            if changed {
                let timestamp_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                let _ = tx.send(CpuPowerSnapshot {
                    timestamp_ms,
                    power_mw: current_power,
                });
            }
        }

        tracing::debug!("CPU power poller sleeping (no subscribers)");
    }
}
