//! GPU management for NVIDIA GPUs using NVML.

use std::time::Instant;
use tokio::sync::mpsc::{Sender, UnboundedReceiver, UnboundedSender};
use tracing::Span;

use crate::error::ZeusdError;

/// A trait for structs that manage one GPU.
///
/// This trait can be used to abstract over different GPU management libraries.
/// Currently, this was done to facilitate testing.
pub trait GpuManager {
    fn device_count() -> Result<u32, ZeusdError>
    where
        Self: Sized;
    fn set_persistent_mode(&mut self, enabled: bool) -> Result<(), ZeusdError>;
    fn set_power_management_limit(&mut self, power_limit: u32) -> Result<(), ZeusdError>;
    fn set_gpu_locked_clocks(
        &mut self,
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    ) -> Result<(), ZeusdError>;
    fn reset_gpu_locked_clocks(&mut self) -> Result<(), ZeusdError>;
    fn set_mem_locked_clocks(
        &mut self,
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    ) -> Result<(), ZeusdError>;
    fn reset_mem_locked_clocks(&mut self) -> Result<(), ZeusdError>;
}

/// A request to execute a GPU command.
///
/// This is the type that is sent to the GPU management background task.
/// The optional `Sender` is used to send a response back to the caller if the
/// user wanted to block until the command is executed.
/// The `Span` is used to propagate tracing context starting from the request.
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
    pub fn start<T>(gpus: Vec<T>) -> anyhow::Result<Self>
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
    /// Enable or disable persistent mode.
    SetPersistentMode { enabled: bool },
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
    fn execute<T>(&self, device: &mut T, request_start_time: Instant) -> Result<(), ZeusdError>
    where
        T: GpuManager,
    {
        match *self {
            Self::SetPersistentMode { enabled } => {
                let result = device.set_persistent_mode(enabled);
                if result.is_ok() {
                    tracing::info!(
                        "Persistent mode {} (took {:?})",
                        if enabled { "enabled" } else { "disabled" },
                        request_start_time.elapsed()
                    );
                } else {
                    tracing::warn!(
                        "Cannot {} persistent mode (took {:?})",
                        if enabled { "enable" } else { "disable" },
                        request_start_time.elapsed()
                    );
                }
                result
            }
            Self::SetPowerLimit {
                power_limit_mw: power_limit,
            } => {
                let result = device.set_power_management_limit(power_limit);
                if result.is_ok() {
                    tracing::info!(
                        "Power limit set to {} W (took {:?})",
                        power_limit / 1000,
                        request_start_time.elapsed()
                    );
                } else {
                    tracing::warn!(
                        "Cannot set power limit to {} W (took {:?}",
                        power_limit / 1000,
                        request_start_time.elapsed()
                    );
                }
                result
            }
            Self::SetGpuLockedClocks {
                min_clock_mhz,
                max_clock_mhz,
            } => {
                let result = device.set_gpu_locked_clocks(min_clock_mhz, max_clock_mhz);
                if result.is_ok() {
                    tracing::info!(
                        "GPU frequency set to [{}, {}] MHz (took {:?})",
                        min_clock_mhz,
                        max_clock_mhz,
                        request_start_time.elapsed()
                    );
                } else {
                    tracing::warn!(
                        "Cannot set GPU frequency to [{}, {}] MHz (took {:?})",
                        min_clock_mhz,
                        max_clock_mhz,
                        request_start_time.elapsed()
                    );
                }
                result
            }
            Self::ResetGpuLockedClocks => {
                let result = device.reset_gpu_locked_clocks();
                if result.is_ok() {
                    tracing::info!(
                        "GPU locked clocks reset (took {:?})",
                        request_start_time.elapsed()
                    );
                } else {
                    tracing::warn!(
                        "Cannot reset GPU locked clocks (took {:?})",
                        request_start_time.elapsed()
                    );
                }
                result
            }
            Self::SetMemLockedClocks {
                min_clock_mhz,
                max_clock_mhz,
            } => {
                let result = device.set_mem_locked_clocks(min_clock_mhz, max_clock_mhz);
                if result.is_ok() {
                    tracing::info!(
                        "Memory locked clocks set to [{}, {}] MHz (took {:?})",
                        min_clock_mhz,
                        max_clock_mhz,
                        request_start_time.elapsed()
                    );
                } else {
                    tracing::warn!(
                        "Cannot set memory locked clocks to [{}, {}] MHz (took {:?})",
                        min_clock_mhz,
                        max_clock_mhz,
                        request_start_time.elapsed()
                    );
                }
                result
            }
            Self::ResetMemLockedClocks => {
                let result = device.reset_mem_locked_clocks();
                if result.is_ok() {
                    tracing::info!(
                        "Memory locked clocks reset (took {:?})",
                        request_start_time.elapsed()
                    );
                } else {
                    tracing::warn!(
                        "Cannot reset memory locked clocks (took {:?})",
                        request_start_time.elapsed()
                    );
                }
                result
            }
        }
    }
}
