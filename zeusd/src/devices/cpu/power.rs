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
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tokio::sync::{watch, Notify};
use tokio::time::{interval, Duration};

use crate::devices::cpu::CpuManager;
use crate::power_streaming::{PowerBroadcast, PowerPoller};

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

/// Broadcast handle for CPU power snapshots.
pub type CpuPowerBroadcast = PowerBroadcast<CpuPowerSnapshot>;

/// Background poller for CPU power.
pub type CpuPowerPoller = PowerPoller<CpuPowerSnapshot>;

/// Start the CPU power polling background task.
///
/// Each CPU must implement `CpuManager`. The poller reads cumulative
/// energy counters and computes instantaneous power from deltas.
/// Polling is demand-driven: sleeps when no subscribers are connected.
pub fn start_cpu_poller<T: CpuManager + Send + 'static>(
    cpus: Vec<(usize, T)>,
    poll_hz: u32,
) -> CpuPowerPoller {
    let valid_ids = cpus.iter().map(|(idx, _)| *idx).collect();
    PowerPoller::start(valid_ids, |tx, subscriber_count, wake| {
        cpu_power_poll_task(cpus, tx, poll_hz, subscriber_count, wake)
    })
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

    tracing::info!(
        "CPU RAPL power poller ready: {} CPUs at {} Hz when subscribers are present",
        cpus.len(),
        poll_hz
    );

    loop {
        // Sleep until at least one subscriber connects.
        while subscriber_count.load(Ordering::Relaxed) == 0 {
            wake.notified().await;
        }

        tracing::info!("CPU power poller starting");

        // Prime energy baselines so the first poll tick computes real deltas.
        let mut states: BTreeMap<usize, CpuEnergyState> = BTreeMap::new();
        for (idx, cpu) in cpus.iter_mut() {
            match cpu.get_cpu_energy() {
                Ok(cpu_energy) => {
                    let dram_energy = if cpu.is_dram_available() {
                        cpu.get_dram_energy().ok()
                    } else {
                        None
                    };
                    states.insert(
                        *idx,
                        CpuEnergyState {
                            last_cpu_energy_uj: cpu_energy,
                            last_dram_energy_uj: dram_energy,
                            last_cpu_power_mw: 0,
                            last_dram_power_mw: if dram_energy.is_some() { Some(0) } else { None },
                        },
                    );
                }
                Err(e) => {
                    tracing::warn!("Failed to prime CPU {} energy baseline: {}", idx, e);
                }
            }
        }

        let mut tick = interval(Duration::from_micros(period_us));
        tick.tick().await; // consume the immediate first tick
        let mut has_broadcast = false;

        // Poll while subscribers are present.
        while subscriber_count.load(Ordering::Relaxed) > 0 {
            tick.tick().await;
            let mut current_power = BTreeMap::new();
            let mut changed = false;

            for (idx, cpu) in cpus.iter_mut() {
                let Some(state) = states.get_mut(idx) else {
                    continue;
                };

                // Read CPU package energy.
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
                        tracing::warn!("Failed to read CPU {} energy: {}", idx, e);
                        state.last_cpu_power_mw
                    }
                };

                // Read DRAM energy if available.
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
                            tracing::warn!("Failed to read CPU {} DRAM energy: {}", idx, e);
                            state.last_dram_power_mw
                        }
                    },
                    None => None,
                };

                current_power.insert(
                    *idx,
                    CpuDramPower {
                        cpu_mw: cpu_power_mw,
                        dram_mw: dram_power_mw,
                    },
                );
            }

            if changed || !has_broadcast {
                has_broadcast = true;
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

        tracing::info!("CPU power poller pausing (no subscribers)");
    }
}
