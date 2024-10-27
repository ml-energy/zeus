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

use serde::Serialize;
use std::path::PathBuf;
use std::sync::RwLock;
use std::time::Instant;
use tokio::sync::mpsc::{Sender, UnboundedReceiver, UnboundedSender};
use tracing::Span;

use crate::error::ZeusdError;

pub struct PackageInfo {
    pub index: u32,
    pub name: String,
    pub energy_uj_path: PathBuf,
    pub max_energy_uj: u64,

    num_wraparounds: RwLock<u64>,
}

#[derive(Serialize)]
pub struct RaplResponse {
    cpu_energy_uj: Option<u64>,
    dram_energy_uj: Option<u64>,
}

pub trait CpuManager {
    fn device_count() -> Result<u32, ZeusdError>;
    fn get_available_fields(index: u32) -> Result<(PackageInfo, Option<PackageInfo>), ZeusdError>;
    fn get_cpu_energy(&self) -> Result<u64, ZeusdError>;
    fn get_dram_energy(&self) -> Result<Option<u64>, ZeusdError>;
    fn stop_monitoring(&mut self);
}

pub type CpuCommandRequest = (
    CpuCommand,
    Option<Sender<Result<RaplResponse, ZeusdError>>>,
    Instant,
    Span,
);

#[derive(Clone)]
pub struct CpuManagementTasks {
    // Senders to the CPU management tasks. index is the CPU ID.
    senders: Vec<UnboundedSender<CpuCommandRequest>>,
}

impl CpuManagementTasks {
    pub fn start<T>(cpus: Vec<T>) -> anyhow::Result<Self>
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
    ) -> Result<RaplResponse, ZeusdError> {
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

    pub async fn stop_monitoring(&self) {
        for sender in self.senders.clone() {
            let (tx, _) = tokio::sync::mpsc::channel(1);
            sender
                .send((
                    CpuCommand::StopMonitoring {},
                    Some(tx),
                    Instant::now(),
                    Span::current(),
                ))
                .unwrap();
        }
    }
}

/// A CPU command that can be executed on a CPU.
#[derive(Debug)]
pub enum CpuCommand {
    /// Get the CPU and DRAM energy measurement for the CPU index
    GetIndexEnergy {
        cpu: bool,
        dram: bool,
    },
    StopMonitoring {},
}

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
    ) -> Result<RaplResponse, ZeusdError>
    where
        T: CpuManager,
    {
        match *self {
            Self::GetIndexEnergy { cpu, dram } => {
                let mut cpu_energy_uj = None;
                let mut dram_energy_uj = None;
                if cpu {
                    cpu_energy_uj = Some(device.get_cpu_energy()?);
                }
                if dram {
                    dram_energy_uj = device.get_dram_energy()?;
                }
                Ok(RaplResponse {
                    cpu_energy_uj,
                    dram_energy_uj,
                })
            }
            Self::StopMonitoring {} => {
                device.stop_monitoring();
                Ok(RaplResponse {
                    cpu_energy_uj: Some(0),
                    dram_energy_uj: Some(0),
                })
            }
        }
    }
}
