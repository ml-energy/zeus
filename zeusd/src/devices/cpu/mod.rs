mod rapl;
pub use rapl::RaplCpu;

pub mod power;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::{Sender, UnboundedReceiver, UnboundedSender};
use tokio::time::{interval, Duration};
use tracing::Span;

use crate::error::ZeusdError;

pub struct PackageInfo {
    pub index: usize,
    pub name: String,
    /// The RAPL powercap zone directory.
    pub zone_dir: PathBuf,
    pub energy_uj_path: PathBuf,
    pub max_energy_uj: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RaplResponse {
    pub cpu_energy_uj: Option<u64>,
    pub dram_energy_uj: Option<u64>,
}

/// One power limit constraint of a RAPL powercap zone.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RaplConstraint {
    /// Constraint name reported by the kernel: `long_term`, `short_term`, or `peak_power`.
    pub name: String,
    pub power_limit_mw: u64,
    /// None if the kernel has no value for this constraint (its sysfs read fails
    /// with `ENODATA`).
    pub max_power_mw: Option<u64>,
    /// None if the kernel has no value for this constraint (its sysfs read fails
    /// with `ENODATA`), which is the case for `peak_power` on kernels 6.5 and later.
    pub time_window_us: Option<u64>,
}

/// Power limit state of one RAPL powercap zone.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RaplZoneLimits {
    /// Whether the kernel reports the zone's `long_term` limit as enabled. The
    /// kernel reports false when the limit is disabled, when it is locked by the
    /// BIOS, or when reading its enable bit failed; sysfs does not distinguish
    /// these cases.
    pub enabled: bool,
    /// Constraints in sysfs index order. Empty if the kernel exposes no
    /// power limits for the zone.
    pub constraints: Vec<RaplConstraint>,
}

/// Power limits of a CPU package zone and its DRAM zone.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RaplPowerLimits {
    pub cpu: RaplZoneLimits,
    pub dram: Option<RaplZoneLimits>,
}

/// Response from a CPU command.
#[derive(Debug)]
pub enum CpuResponse {
    Energy(RaplResponse),
    PowerLimits(RaplPowerLimits),
}

pub trait CpuManager {
    /// Get the number of CPUs available.
    fn device_count() -> Result<usize, ZeusdError>;
    /// Get the CPU PackageInfo and the DRAM PackageInfo it is available.
    fn get_available_fields(
        index: usize,
    ) -> Result<(Arc<PackageInfo>, Option<Arc<PackageInfo>>), ZeusdError>;
    /// Get the cumulative energy counter value of the CPU after compensating for wraparounds.
    fn get_cpu_energy(&mut self) -> Result<u64, ZeusdError>;
    /// Get the cumulative energy counter value of the DRAM after compensating for wraparounds.
    fn get_dram_energy(&mut self) -> Result<u64, ZeusdError>;
    /// Check if DRAM is available.
    fn is_dram_available(&self) -> bool;
    /// Read the power limits of the CPU package zone and, if available, the DRAM zone.
    fn get_power_limits(&self) -> Result<RaplPowerLimits, ZeusdError>;
}

pub type CpuCommandRequest = (
    CpuCommand,
    Option<Sender<Result<CpuResponse, ZeusdError>>>,
    Instant,
    Span,
);

#[derive(Clone)]
pub struct CpuManagementTasks {
    // Senders to the CPU management tasks. index is the CPU ID.
    senders: Vec<UnboundedSender<CpuCommandRequest>>,
}

impl CpuManagementTasks {
    pub fn start<T>(cpus: Vec<T>) -> Result<Self, ZeusdError>
    where
        T: CpuManager + Send + 'static,
    {
        let mut senders = Vec::with_capacity(cpus.len());
        for (cpu_id, cpu) in cpus.into_iter().enumerate() {
            // Channel to send commands to the CPU management task.
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            senders.push(tx);
            // The CPU management task will automatically terminate
            // when the server terminates and the last sender is dropped.
            tokio::spawn(cpu_management_task(cpu, rx));
            tracing::info!("Background task for CPU {} successfully spawned", cpu_id);
        }
        Ok(Self { senders })
    }

    /// Return the number of CPUs managed by these tasks.
    pub fn device_count(&self) -> usize {
        self.senders.len()
    }

    pub async fn send_command_blocking(
        &self,
        cpu_id: usize,
        command: CpuCommand,
        request_start_time: Instant,
    ) -> Result<CpuResponse, ZeusdError> {
        if cpu_id >= self.senders.len() {
            return Err(ZeusdError::CpuNotFoundError(cpu_id));
        }
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        self.senders[cpu_id]
            .send((command, Some(tx), request_start_time, Span::current()))
            .map_err(ZeusdError::from)?;
        match rx.recv().await {
            Some(result) => result,
            None => Err(ZeusdError::CpuManagementTaskTerminatedError(cpu_id)),
        }
    }
}

/// A CPU command that can be executed on a CPU.
#[derive(Debug)]
pub enum CpuCommand {
    /// Get the CPU and DRAM energy measurement for the CPU index.
    GetIndexEnergy { cpu: bool, dram: bool },
    /// Get the power limits of the CPU package and DRAM zones.
    GetPowerLimits,
}

/// Tokio background task that handles requests to each CPU.
///
/// Between commands, a periodic keepalive reads the energy counters every 30
/// seconds so that RAPL counter wraparounds are detected even during idle
/// periods when no client is actively querying energy.
async fn cpu_management_task<T: CpuManager>(
    mut cpu: T,
    mut rx: UnboundedReceiver<CpuCommandRequest>,
) {
    let mut keepalive = interval(Duration::from_secs(30));
    // The first tick completes immediately; consume it so we don't
    // do a spurious keepalive read right at startup.
    keepalive.tick().await;

    loop {
        tokio::select! {
            Some((command, response, start_time, span)) = rx.recv() => {
                let _span_guard = span.enter();
                let result = command.execute(&mut cpu, start_time);
                if let Some(response) = response {
                    if response.send(result).await.is_err() {
                        tracing::error!("Failed to send response to caller");
                    }
                }
            }
            _ = keepalive.tick() => {
                let _ = cpu.get_cpu_energy();
                if cpu.is_dram_available() {
                    let _ = cpu.get_dram_energy();
                }
            }
            else => break,
        }
    }
}

impl CpuCommand {
    fn execute<T>(
        &self,
        device: &mut T,
        _request_arrival_time: Instant,
    ) -> Result<CpuResponse, ZeusdError>
    where
        T: CpuManager,
    {
        match *self {
            Self::GetIndexEnergy { cpu, dram } => {
                let cpu_energy_uj = if cpu {
                    Some(device.get_cpu_energy()?)
                } else {
                    None
                };
                let dram_energy_uj = if dram && device.is_dram_available() {
                    Some(device.get_dram_energy()?)
                } else {
                    None
                };
                Ok(CpuResponse::Energy(RaplResponse {
                    cpu_energy_uj,
                    dram_energy_uj,
                }))
            }
            Self::GetPowerLimits => device.get_power_limits().map(CpuResponse::PowerLimits),
        }
    }
}
