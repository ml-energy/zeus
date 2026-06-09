//! CPU RAPL power polling and streaming.
//!
//! A dedicated background task polls CPU package and DRAM energy counters
//! at a configurable frequency, computes instantaneous power from energy
//! deltas, and broadcasts snapshots via a `tokio::sync::watch` channel.
//! Polling is demand-driven: the task sleeps when no subscribers are
//! connected and wakes when the first subscriber arrives.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use serde::Serialize;
use tokio::sync::{watch, Notify};
use tokio::time::{interval, Duration};

use crate::devices::cpu::CpuManager;
use crate::power_streaming::{unix_timestamp_ms, PowerBroadcast, PowerBroadcasts, PowerPoller};

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
    pub power_mw: BTreeMap<usize, CpuDramPower>,
}

/// A single CPU power sample for SSE streaming.
#[derive(Clone, Debug, Default, Serialize)]
pub struct CpuPowerSample {
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// CPU index.
    pub cpu_id: usize,
    /// CPU package power in milliwatts.
    pub cpu_mw: u32,
    /// DRAM power in milliwatts, if available.
    pub dram_mw: Option<u32>,
}

/// Broadcast handle for CPU power samples.
pub type CpuPowerBroadcast = PowerBroadcast<CpuPowerSample>;

/// CPU power broadcasts keyed by CPU index.
pub type CpuPowerBroadcasts = PowerBroadcasts<CpuPowerSample>;

/// Background poller for one CPU's power.
pub type CpuPowerPoller = PowerPoller<CpuPowerSample>;

/// Start the CPU power polling background task.
///
/// Each CPU must implement `CpuManager`. The poller reads cumulative
/// energy counters and computes instantaneous power from deltas.
/// Polling is demand-driven: sleeps when no subscribers are connected.
pub fn start_cpu_poller<T: CpuManager + Send + 'static>(
    cpus: Vec<(usize, T)>,
    poll_hz: u32,
) -> CpuPowerBroadcasts {
    let mut broadcasts = BTreeMap::new();
    for (cpu_id, cpu) in cpus {
        let poller = PowerPoller::start(move |tx, subscriber_count, wake| {
            cpu_power_poll_task(cpu_id, cpu, tx, poll_hz, subscriber_count, wake)
        });
        broadcasts.insert(cpu_id, poller.broadcast());
    }
    PowerBroadcasts::new(broadcasts)
}

/// Per-CPU energy tracking state for computing power from energy deltas.
struct CpuEnergyState {
    last_cpu_energy_uj: u64,
    /// `None` means DRAM is not available for this CPU.
    last_dram_energy_uj: Option<u64>,
    last_cpu_power_mw: u32,
    last_dram_power_mw: Option<u32>,
}

async fn cpu_power_poll_task<T: CpuManager>(
    cpu_id: usize,
    mut cpu: T,
    tx: watch::Sender<CpuPowerSample>,
    poll_hz: u32,
    subscriber_count: Arc<AtomicUsize>,
    wake: Arc<Notify>,
) {
    let period_us = 1_000_000u64 / poll_hz.max(1) as u64;

    tracing::info!(
        "CPU RAPL power poller ready for CPU {} at {} Hz when subscribers are present",
        cpu_id,
        poll_hz
    );

    'poller: loop {
        // Sleep until at least one subscriber connects.
        while subscriber_count.load(Ordering::Relaxed) == 0 {
            wake.notified().await;
        }

        tracing::info!("CPU power poller starting for CPU {}", cpu_id);

        let mut state = loop {
            match cpu.get_cpu_energy() {
                Ok(cpu_energy) => {
                    let dram_energy = if cpu.is_dram_available() {
                        match cpu.get_dram_energy() {
                            Ok(energy) => Some(energy),
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to prime CPU {} DRAM energy baseline: {}",
                                    cpu_id,
                                    e
                                );
                                None
                            }
                        }
                    } else {
                        None
                    };
                    break CpuEnergyState {
                        last_cpu_energy_uj: cpu_energy,
                        last_dram_energy_uj: dram_energy,
                        last_cpu_power_mw: 0,
                        last_dram_power_mw: if dram_energy.is_some() { Some(0) } else { None },
                    };
                }
                Err(e) => {
                    tracing::warn!("Failed to prime CPU {} energy baseline: {}", cpu_id, e);
                    if subscriber_count.load(Ordering::Relaxed) == 0 {
                        continue 'poller;
                    }
                    tokio::time::sleep(Duration::from_micros(period_us)).await;
                }
            }
        };

        let mut tick = interval(Duration::from_micros(period_us));
        tick.tick().await; // consume the immediate first tick
        let mut has_broadcast = false;

        // Poll while subscribers are present.
        while subscriber_count.load(Ordering::Relaxed) > 0 {
            tick.tick().await;
            let mut changed = false;

            let cpu_power_mw = match cpu.get_cpu_energy() {
                Ok(energy_uj) => {
                    let delta_uj = energy_uj.saturating_sub(state.last_cpu_energy_uj);
                    let power_mw = (delta_uj * 1000 / period_us) as u32;
                    if power_mw != state.last_cpu_power_mw {
                        changed = true;
                    }
                    state.last_cpu_power_mw = power_mw;
                    state.last_cpu_energy_uj = energy_uj;
                    power_mw
                }
                Err(e) => {
                    tracing::warn!("Failed to read CPU {} energy: {}", cpu_id, e);
                    state.last_cpu_power_mw
                }
            };

            let dram_power_mw = match state.last_dram_energy_uj {
                Some(last_dram) => match cpu.get_dram_energy() {
                    Ok(energy_uj) => {
                        let delta_uj = energy_uj.saturating_sub(last_dram);
                        let power_mw = (delta_uj * 1000 / period_us) as u32;
                        if state.last_dram_power_mw != Some(power_mw) {
                            changed = true;
                        }
                        state.last_dram_power_mw = Some(power_mw);
                        state.last_dram_energy_uj = Some(energy_uj);
                        Some(power_mw)
                    }
                    Err(e) => {
                        tracing::warn!("Failed to read CPU {} DRAM energy: {}", cpu_id, e);
                        state.last_dram_power_mw
                    }
                },
                None => None,
            };

            if changed || !has_broadcast {
                has_broadcast = true;
                let _ = tx.send(CpuPowerSample {
                    timestamp_ms: unix_timestamp_ms(),
                    cpu_id,
                    cpu_mw: cpu_power_mw,
                    dram_mw: dram_power_mw,
                });
            }
        }

        tracing::info!("CPU power poller pausing for CPU {}", cpu_id);
    }
}
