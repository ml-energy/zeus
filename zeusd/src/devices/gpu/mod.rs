//! GPU management

// NVIDIA GPU.
// Real NVML interface.
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::NvmlGpu;

// Fake NVML interface for dev and testing on macOS.
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub use macos::NvmlGpu;

// GPU management.
// As long as there is a struct that implements the GpuManager trait,
// the code below will work with any GPU management library.
use std::time::Instant;
use tokio::sync::mpsc::{Sender, UnboundedReceiver, UnboundedSender};
use tracing::Span;

use crate::error::ZeusdError;

/// A trait for structs that manage one GPU.
///
/// This trait can be used to abstract over different GPU management libraries.
/// Currently, this was done to facilitate testing.
pub trait GpuManager {
    /// Get the number of GPUs visible in the node.
    fn device_count() -> Result<u32, ZeusdError>
    where
        Self: Sized;
    /// Set the persistence mode of the GPU.
    fn set_persistence_mode(&mut self, enabled: bool) -> Result<(), ZeusdError>;
    /// Set the power management limit in milliwatts.
    fn set_power_management_limit(&mut self, power_limit: u32) -> Result<(), ZeusdError>;
    /// Set the GPU's locked clock range in MHz.
    fn set_gpu_locked_clocks(
        &mut self,
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    ) -> Result<(), ZeusdError>;
    /// Reset the GPU's locked clocks.
    fn reset_gpu_locked_clocks(&mut self) -> Result<(), ZeusdError>;
    /// Set the memory locked clock range in MHz.
    fn set_mem_locked_clocks(
        &mut self,
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    ) -> Result<(), ZeusdError>;
    /// Reset the memory locked clocks.
    fn reset_mem_locked_clocks(&mut self) -> Result<(), ZeusdError>;
}

/// A request to execute a GPU command.
///
/// This is the type that is sent to the GPU management background task.
/// The optional `Sender` is used to send a response back to the caller if the
/// user wanted to block until the command is done executing.
/// The `Instant` object is when the request was received by the server.
/// It's used to log how long it took until the command was executed on the GPU.
/// The `Span` object is used to propagate tracing context starting from the request.
pub type GpuCommandRequest = (
    GpuCommand,
    Option<Sender<Result<(), ZeusdError>>>,
    Instant,
    Span,
);

/// A collection of GPU management tasks.
///
/// This struct is used to send commands to the GPU management tasks.
/// It's also application state that gets cloned and passed to request handlers by actix-web.
#[derive(Clone)]
pub struct GpuManagementTasks {
    // Senders to the GPU management tasks. index is the GPU ID.
    senders: Vec<UnboundedSender<GpuCommandRequest>>,
}

impl GpuManagementTasks {
    /// Start GPU management tasks for the given GPUs.
    /// It's generic over the type of GPU manager to allow for testing.
    pub fn start<T>(gpus: Vec<T>) -> Result<Self, ZeusdError>
    where
        T: GpuManager + Send + 'static,
    {
        let mut senders = Vec::with_capacity(gpus.len());
        for (gpu_id, gpu) in gpus.into_iter().enumerate() {
            // Channel to send commands to the GPU management task.
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            senders.push(tx);
            // The GPU management task will automatically terminate
            // when the server terminates and the last sender is dropped.
            tokio::spawn(gpu_management_task(gpu, rx));
            tracing::info!("Background task for GPU {} successfully spawned", gpu_id);
        }
        Ok(Self { senders })
    }

    /// Send a command to the corresponding GPU management task and immediately return
    /// without checking the result. Results will be logged via tracing.
    /// Returns `Ok(())` if the command was *sent* successfully.
    pub fn send_command_nonblocking(
        &self,
        gpu_id: usize,
        command: GpuCommand,
        request_start_time: Instant,
    ) -> Result<(), ZeusdError> {
        if gpu_id >= self.senders.len() {
            return Err(ZeusdError::GpuNotFoundError(gpu_id));
        }
        self.senders[gpu_id]
            .send((command, None, request_start_time, Span::current()))
            .map_err(|e| e.into())
    }

    /// Send a command to the corresponding GPU management task and wait for completion.
    /// Returns `Ok(())` if the command was *executed* successfully.
    pub async fn send_command_blocking(
        &self,
        gpu_id: usize,
        command: GpuCommand,
        request_start_time: Instant,
    ) -> Result<(), ZeusdError> {
        if gpu_id >= self.senders.len() {
            return Err(ZeusdError::GpuNotFoundError(gpu_id));
        }
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        self.senders[gpu_id]
            .send((command, Some(tx), request_start_time, Span::current()))
            .map_err(ZeusdError::from)?;
        match rx.recv().await {
            Some(result) => result,
            None => Err(ZeusdError::GpuManagementTaskTerminatedError(gpu_id)),
        }
    }
}

/// A asynchronous Tokio background task that manages one GPU.
///
/// Listens for commands on a channel and executes them on the GPU it manages.
async fn gpu_management_task<T: GpuManager>(
    mut gpu: T,
    mut rx: UnboundedReceiver<GpuCommandRequest>,
) {
    while let Some((command, response, start_time, span)) = rx.recv().await {
        let _span_guard = span.enter();
        let result = command.execute(&mut gpu, start_time);
        if let Some(response) = response {
            if response.send(result).await.is_err() {
                tracing::error!("Failed to send response to caller");
            }
        }
    }
}

/// A GPU command that can be executed on a GPU.
#[derive(Debug)]
pub enum GpuCommand {
    /// Enable or disable persistence mode.
    SetPersistenceMode { enabled: bool },
    /// Set the power management limit in milliwatts.
    SetPowerLimit { power_limit_mw: u32 },
    /// Set the GPU's locked clock range in MHz.
    SetGpuLockedClocks {
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    },
    /// Reset the GPU's locked clocks.
    ResetGpuLockedClocks,
    /// Set the GPU's memory locked clock range in MHz.
    SetMemLockedClocks {
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    },
    /// Reset the GPU's memory locked clocks.
    ResetMemLockedClocks,
}

impl GpuCommand {
    fn execute<T>(&self, device: &mut T, request_arrival_time: Instant) -> Result<(), ZeusdError>
    where
        T: GpuManager,
    {
        match *self {
            Self::SetPersistenceMode { enabled } => {
                let command_start_time = Instant::now();
                let result = device.set_persistence_mode(enabled);
                if result.is_ok() {
                    tracing::info!(
                        time_to_command_done = ?request_arrival_time.elapsed(),
                        zeusd_overhead = ?command_start_time - request_arrival_time,
                        "Persistence mode {}",
                        if enabled { "enabled" } else { "disabled" },
                    );
                } else {
                    tracing::warn!(
                        time_to_command_done = ?request_arrival_time.elapsed(),
                        zeusd_overhead = ?command_start_time - request_arrival_time,
                        "Cannot {} persistence mode",
                        if enabled { "enable" } else { "disable" },
                    );
                }
                result
            }
            Self::SetPowerLimit {
                power_limit_mw: power_limit,
            } => {
                let command_start_time = Instant::now();
                let result = device.set_power_management_limit(power_limit);
                if result.is_ok() {
                    tracing::info!(
                        time_to_command_done = ?request_arrival_time.elapsed(),
                        zeusd_overhead = ?command_start_time - request_arrival_time,
                        "Power limit set to {} W",
                        power_limit / 1000,
                    );
                } else {
                    tracing::warn!(
                        time_to_command_done = ?request_arrival_time.elapsed(),
                        zeusd_overhead = ?command_start_time - request_arrival_time,
                        "Cannot set power limit to {} W ",
                        power_limit / 1000,
                    );
                }
                result
            }
            Self::SetGpuLockedClocks {
                min_clock_mhz,
                max_clock_mhz,
            } => {
                let command_start_time = Instant::now();
                let result = device.set_gpu_locked_clocks(min_clock_mhz, max_clock_mhz);
                if result.is_ok() {
                    tracing::info!(
                        time_to_command_done = ?request_arrival_time.elapsed(),
                        zeusd_overhead = ?command_start_time - request_arrival_time,
                        "GPU frequency set to [{}, {}] MHz",
                        min_clock_mhz,
                        max_clock_mhz,
                    );
                } else {
                    tracing::warn!(
                        time_to_command_done = ?request_arrival_time.elapsed(),
                        zeusd_overhead = ?command_start_time - request_arrival_time,
                        "Cannot set GPU frequency to [{}, {}] MHz",
                        min_clock_mhz,
                        max_clock_mhz,
                    );
                }
                result
            }
            Self::ResetGpuLockedClocks => {
                let command_start_time = Instant::now();
                let result = device.reset_gpu_locked_clocks();
                if result.is_ok() {
                    tracing::info!(
                        time_to_command_done = ?request_arrival_time.elapsed(),
                        zeusd_overhead = ?command_start_time - request_arrival_time,
                        "GPU locked clocks reset",
                    );
                } else {
                    tracing::warn!(
                        time_to_command_done = ?request_arrival_time.elapsed(),
                        zeusd_overhead = ?command_start_time - request_arrival_time,
                        "Cannot reset GPU locked clocks",
                    );
                }
                result
            }
            Self::SetMemLockedClocks {
                min_clock_mhz,
                max_clock_mhz,
            } => {
                let command_start_time = Instant::now();
                let result = device.set_mem_locked_clocks(min_clock_mhz, max_clock_mhz);
                if result.is_ok() {
                    tracing::info!(
                        time_to_command_done = ?request_arrival_time.elapsed(),
                        zeusd_overhead = ?command_start_time - request_arrival_time,
                        "Memory locked clocks set to [{}, {}] MHz",
                        min_clock_mhz,
                        max_clock_mhz,
                    );
                } else {
                    tracing::warn!(
                        time_to_command_done = ?request_arrival_time.elapsed(),
                        zeusd_overhead = ?command_start_time - request_arrival_time,
                        "Cannot set memory locked clocks to [{}, {}] MHz",
                        min_clock_mhz,
                        max_clock_mhz,
                    );
                }
                result
            }
            Self::ResetMemLockedClocks => {
                let command_start_time = Instant::now();
                let result = device.reset_mem_locked_clocks();
                if result.is_ok() {
                    tracing::info!(
                        time_to_command_done = ?request_arrival_time.elapsed(),
                        zeusd_overhead = ?command_start_time - request_arrival_time,
                        "Memory locked clocks reset",
                    );
                } else {
                    tracing::warn!(
                        time_to_command_done = ?request_arrival_time.elapsed(),
                        zeusd_overhead = ?command_start_time - request_arrival_time,
                        "Cannot reset memory locked clocks",
                    );
                }
                result
            }
        }
    }
}
