//! GPU management module that interfaces with NVML

use nvml_wrapper::enums::device::GpuLockedClocksSetting;
use nvml_wrapper::{Device, Nvml};
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

pub struct NvmlGpu<'n> {
    _nvml: &'static Nvml,
    device: Device<'n>,
}

impl NvmlGpu<'static> {
    pub fn init(index: u32) -> Result<Self, ZeusdError> {
        // `Device` needs to hold a reference to `Nvml`, meaning that `Nvml` must outlive `Device`.
        // We can achieve this by leaking a `Box` containing `Nvml` and holding a reference to it.
        // `Nvml` will actually live until the server terminates inside the GPU management task.
        let _nvml = Box::leak(Box::new(Nvml::init()?));
        let device = _nvml.device_by_index(index)?;
        Ok(Self { _nvml, device })
    }
}

impl GpuManager for NvmlGpu<'static> {
    fn device_count() -> Result<u32, ZeusdError> {
        let nvml = Nvml::init()?;
        Ok(nvml.device_count()?)
    }

    #[inline]
    fn set_persistent_mode(&mut self, enabled: bool) -> Result<(), ZeusdError> {
        Ok(self.device.set_persistent(enabled)?)
    }

    #[inline]
    fn set_power_management_limit(&mut self, power_limit_mw: u32) -> Result<(), ZeusdError> {
        Ok(self.device.set_power_management_limit(power_limit_mw)?)
    }

    #[inline]
    fn set_gpu_locked_clocks(
        &mut self,
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    ) -> Result<(), ZeusdError> {
        let setting = GpuLockedClocksSetting::Numeric {
            min_clock_mhz,
            max_clock_mhz,
        };
        Ok(self.device.set_gpu_locked_clocks(setting)?)
    }

    #[inline]
    fn reset_gpu_locked_clocks(&mut self) -> Result<(), ZeusdError> {
        Ok(self.device.reset_gpu_locked_clocks()?)
    }

    #[inline]
    fn set_mem_locked_clocks(
        &mut self,
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    ) -> Result<(), ZeusdError> {
        Ok(self
            .device
            .set_mem_locked_clocks(min_clock_mhz, max_clock_mhz)?)
    }

    #[inline]
    fn reset_mem_locked_clocks(&mut self) -> Result<(), ZeusdError> {
        Ok(self.device.reset_mem_locked_clocks()?)
    }
}

/// A request to execute a GPU command.
///
/// This is the type that is sent to the GPU management background task.
/// The optional `Sender` is used to send a response back to the caller if the
/// user wanted to block until the command is executed.
/// The `Span` is used to propagate tracing context starting from the request.
pub type GpuCommandRequest = (GpuCommand, Option<Sender<Result<(), ZeusdError>>>, Span);

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
    ) -> Result<(), ZeusdError> {
        if gpu_id >= self.senders.len() {
            return Err(ZeusdError::GpuNotFoundError(gpu_id));
        }
        self.senders[gpu_id]
            .send((command, None, Span::current()))
            .map_err(|e| e.into())
    }

    /// Send a command to the corresponding GPU management task and wait for completion.
    /// Returns `Ok(())` if the command was *executed* successfully.
    pub async fn send_command_blocking(
        &self,
        gpu_id: usize,
        command: GpuCommand,
    ) -> Result<(), ZeusdError> {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        self.senders[gpu_id]
            .send((command, Some(tx), Span::current()))
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
    while let Some((command, response, span)) = rx.recv().await {
        let _span_guard = span.enter();
        let result = command.execute(&mut gpu);
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
    fn execute<T>(&self, device: &mut T) -> Result<(), ZeusdError>
    where
        T: GpuManager,
    {
        match *self {
            Self::SetPersistentMode { enabled } => {
                let result = device.set_persistent_mode(enabled);
                if result.is_ok() {
                    tracing::info!(
                        "Persistent mode {}",
                        if enabled { "enabled" } else { "disabled" }
                    );
                } else {
                    tracing::warn!(
                        "Cannot {} persistent mode",
                        if enabled { "enable" } else { "disable" }
                    );
                }
                result
            }
            Self::SetPowerLimit {
                power_limit_mw: power_limit,
            } => {
                let result = device.set_power_management_limit(power_limit);
                if result.is_ok() {
                    tracing::info!("Power limit set to {} W", power_limit / 1000);
                } else {
                    tracing::warn!("Cannot set power limit to {} W", power_limit / 1000);
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
                        "GPU frequency set to [{}, {}] MHz",
                        min_clock_mhz,
                        max_clock_mhz
                    );
                } else {
                    tracing::warn!(
                        "Cannot set GPU frequency to [{}, {}] MHz",
                        min_clock_mhz,
                        max_clock_mhz
                    );
                }
                result
            }
            Self::ResetGpuLockedClocks => {
                let result = device.reset_gpu_locked_clocks();
                if result.is_ok() {
                    tracing::info!("GPU locked clocks reset");
                } else {
                    tracing::warn!("Cannot reset GPU locked clocks");
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
                        "Memory locked clocks set to [{}, {}] MHz",
                        min_clock_mhz,
                        max_clock_mhz,
                    );
                } else {
                    tracing::warn!(
                        "Cannot set memory locked clocks to [{}, {}] MHz",
                        min_clock_mhz,
                        max_clock_mhz,
                    );
                }
                result
            }
            Self::ResetMemLockedClocks => {
                let result = device.reset_mem_locked_clocks();
                if result.is_ok() {
                    tracing::info!("Memory locked clocks reset");
                } else {
                    tracing::warn!("Cannot reset memory locked clocks");
                }
                result
            }
        }
    }
}
