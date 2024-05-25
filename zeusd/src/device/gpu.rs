use crate::error::ZeusdError;
use anyhow::Context;
use nvml_wrapper::enums::device::GpuLockedClocksSetting;
use nvml_wrapper::Nvml;
use tokio::sync::mpsc::{Sender, UnboundedReceiver, UnboundedSender};

#[derive(Clone, Debug)]
pub struct GpuHandlers {
    senders: Vec<UnboundedSender<(GpuCommand, Option<Sender<Result<(), ZeusdError>>>)>>,
}

impl GpuHandlers {
    pub async fn start() -> anyhow::Result<Self> {
        let nvml = Nvml::init()?;
        let num_gpus = nvml.device_count()?;
        let mut senders = Vec::with_capacity(num_gpus as usize);

        for gpu_id in 0..num_gpus {
            // Channel to send commands to the GPU handler.
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            senders.push(tx);
            // Channel to receive the initialization response from the GPU handler.
            let (init_tx, mut init_rx) = tokio::sync::mpsc::channel(1);
            // The GPU handler background task will automatically terminate
            // when the server terminates and the last sender is dropped.
            tokio::spawn(gpu_handler(gpu_id, init_tx, rx));
            let init_message = init_rx
                .recv()
                .await
                .with_context(|| {
                    format!(
                        "Failed to receive initialization response from GPU {}",
                        gpu_id
                    )
                })?
                .with_context(|| format!("Failed to initialize handler for GPU {}", gpu_id))?;
            tracing::info!("GPU {} handler initialized: {}", gpu_id, init_message);
        }
        Ok(Self { senders })
    }

    pub fn send_command_nonblocking(
        &self,
        gpu_id: usize,
        command: GpuCommand,
    ) -> Result<(), ZeusdError> {
        if gpu_id >= self.senders.len() {
            return Err(ZeusdError::GpuNotFoundError(gpu_id));
        }
        self.senders[gpu_id]
            .send((command, None))
            .map_err(|e| e.into())
    }

    pub async fn send_command_block(
        &self,
        gpu_id: usize,
        command: GpuCommand,
    ) -> Result<(), ZeusdError> {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        self.senders[gpu_id]
            .send((command, Some(tx)))
            .map_err(|e| ZeusdError::from(e))?;
        match rx.recv().await {
            Some(result) => result.map_err(|e| e.into()),
            None => Err(ZeusdError::GpuHandlerTerminatedError(gpu_id)),
        }
    }
}

fn handle_gpu_command(
    device: &mut nvml_wrapper::Device,
    command: GpuCommand,
) -> Result<(), nvml_wrapper::error::NvmlError> {
    match command {
        GpuCommand::SetPersistentMode { enabled } => device.set_persistent(enabled),
        GpuCommand::SetPowerLimit { power_limit } => device.set_power_management_limit(power_limit),
        GpuCommand::SetGpuFrequency {
            max_frequency,
            min_frequency,
        } => {
            let clock_settings = GpuLockedClocksSetting::Numeric {
                min_clock_mhz: min_frequency,
                max_clock_mhz: max_frequency,
            };
            device.set_gpu_locked_clocks(clock_settings)
        }
    }
}

async fn gpu_handler(
    gpu_id: u32,
    init_tx: Sender<anyhow::Result<String>>,
    mut rx: UnboundedReceiver<(GpuCommand, Option<Sender<Result<(), ZeusdError>>>)>,
) {
    match Nvml::init() {
        Ok(nvml) => match nvml.device_by_index(gpu_id) {
            Ok(mut device) => {
                if init_tx
                    .send(Ok(format!("GPU {} handler initialized", gpu_id)))
                    .await
                    .is_err()
                {
                    tracing::error!(
                        "Failed to send initialization response. Terminating GPU {} handler.",
                        gpu_id,
                    );
                    return;
                }
                while let Some((command, response)) = rx.recv().await {
                    let result = handle_gpu_command(&mut device, command);
                    if let Some(response) = response {
                        if response.send(result.map_err(|e| e.into())).await.is_err() {
                            tracing::error!("Failed to send response to caller");
                        }
                    }
                }
            }
            Err(e) => {
                let _ = init_tx
                    .send(Err(e).with_context(|| {
                        format!("Failed to get device handle of GPU {} from NVML.", gpu_id)
                    }))
                    .await;
                return;
            }
        },
        Err(e) => {
            let _ = init_tx
                .send(Err(e).with_context(|| {
                    format!("Failed to initialize NVML from GPU {}'s handler.", gpu_id)
                }))
                .await;
            return;
        }
    }
}

#[derive(Debug)]
pub enum GpuCommand {
    SetPersistentMode {
        enabled: bool,
    },
    SetPowerLimit {
        power_limit: u32,
    },
    SetGpuFrequency {
        max_frequency: u32,
        min_frequency: u32,
    },
}
