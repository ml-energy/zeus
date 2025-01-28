// RAPL CPU
// Real RAPL interface.
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::RaplCpu;

// Fake Rapl interface for dev and testing on macOS.
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub use macos::RaplCpu;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Instant;
use tokio::sync::mpsc::{Sender, UnboundedReceiver, UnboundedSender};
use tracing::Span;

use crate::error::ZeusdError;

pub struct PackageInfo {
    pub index: usize,
    pub name: String,
    pub energy_uj_path: PathBuf,
    pub max_energy_uj: u64,
    pub num_wraparounds: RwLock<u64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RaplResponse {
    pub cpu_energy_uj: Option<u64>,
    pub dram_energy_uj: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DramAvailabilityResponse {
    pub dram_available: bool,
}

/// Unified CPU response type
#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum CpuResponse {
    Rapl(RaplResponse),
    Dram(DramAvailabilityResponse),
}

pub trait CpuManager {
    /// Get the number of CPUs available.
    fn device_count() -> Result<usize, ZeusdError>;
    /// Get the CPU PackageInfo and the DRAM PackageInfo it is available.
    fn get_available_fields(
        index: usize,
    ) -> Result<(Arc<PackageInfo>, Option<Arc<PackageInfo>>), ZeusdError>;
    // Get the cumulative Rapl count value of the CPU after compensating for wraparounds.
    fn get_cpu_energy(&mut self) -> Result<u64, ZeusdError>;
    // Get the cumulative Rapl count value of the DRAM after compensating for wraparounds if it is
    // available.
    fn get_dram_energy(&mut self) -> Result<u64, ZeusdError>;
    // Abort the monitoring tasks for CPU and DRAM if the tasks have been started.
    fn stop_monitoring(&mut self);
    // Check if DRAM is available.
    fn is_dram_available(&self) -> bool;
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
            .unwrap();
        match rx.recv().await {
            Some(result) => result,
            None => Err(ZeusdError::CpuManagementTaskTerminatedError(cpu_id)),
        }
    }

    pub async fn stop_monitoring(&self) -> Result<(), ZeusdError> {
        for (index, sender) in self.senders.iter().enumerate() {
            let (tx, mut rx) = tokio::sync::mpsc::channel(1);
            sender
                .send((
                    CpuCommand::StopMonitoring,
                    Some(tx),
                    Instant::now(),
                    Span::current(),
                ))
                .unwrap();
            match rx.recv().await {
                Some(_) => {}
                None => return Err(ZeusdError::CpuManagementTaskTerminatedError(index)),
            }
        }
        Ok(())
    }
}

/// A CPU command that can be executed on a CPU.
#[derive(Debug)]
pub enum CpuCommand {
    /// Get the CPU and DRAM energy measurement for the CPU index
    GetIndexEnergy { cpu: bool, dram: bool },
    /// Return if the specified CPU supports DRAM energy measurement
    SupportsDramEnergy,
    /// Stop the monitoring task for CPU and DRAM if they have been started.
    StopMonitoring,
}

/// Tokio background task that handles requests to each CPU.
/// NOTE: Currently, this serializes the handling of request to a single CPU, which is
///       largely unnecessary as the requests are simply reading energy counters.
///       This is subject to refactoring if it is to become a bottleneck.
async fn cpu_management_task<T: CpuManager>(
    mut cpu: T,
    mut rx: UnboundedReceiver<CpuCommandRequest>,
) {
    while let Some((command, response, start_time, span)) = rx.recv().await {
        let _span_guard = span.enter();
        let result = command.execute(&mut cpu, start_time);
        if let Some(response) = response {
            if response.send(result).await.is_err() {
                tracing::error!("Failed to send response to caller");
            }
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
                // Wrap the RaplResponse in CpuResponse::Rapl
                Ok(CpuResponse::Rapl(RaplResponse {
                    cpu_energy_uj,
                    dram_energy_uj,
                }))
            }
            Self::SupportsDramEnergy => {
                // Wrap the DramAvailabilityResponse in CpuResponse::Dram
                Ok(CpuResponse::Dram(DramAvailabilityResponse {
                    dram_available: device.is_dram_available(),
                }))
            }
            Self::StopMonitoring => {
                device.stop_monitoring();
                Ok(CpuResponse::Rapl(RaplResponse {
                    cpu_energy_uj: Some(0),
                    dram_energy_uj: Some(0),
                }))
            }
        }
    }
}
