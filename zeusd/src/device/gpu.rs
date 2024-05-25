use tokio::sync::mpsc::{Sender, UnboundedSender};

use crate::error::ZeusdError;

#[derive(Clone, Debug)]
pub struct GpuHandlers {
    senders: Vec<UnboundedSender<GpuCommand>>,
}

impl GpuHandlers {
    pub fn start() -> Self {
        let senders = vec![];
        Self { senders }
    }

    pub fn send_command(&self, gpu_id: usize, command: GpuCommand) {
        self.senders[gpu_id].send(command).unwrap();
    }
}

#[derive(Debug)]
pub enum GpuCommand {
    SetPowerLimit {
        power_limit: usize,
        response: Option<Sender<Result<(), ZeusdError>>>,
    },
    SetFrequency {
        frequency: usize,
        response: Option<Sender<Result<(), ZeusdError>>>,
    },
}
