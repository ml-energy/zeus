//! One native command thread and one ordered queue per device.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use tokio::sync::mpsc::{error::SendError, Sender};
use tracing::Span;

use crate::error::ZeusdError;
use crate::operations::{OperationError, OperationTicket};

/// Includes cancelled entries waiting behind an uninterruptible native call.
const MAX_TRACKED_QUEUE: usize = 1024;

pub type Request<C, R> = (C, Option<Sender<Result<R, ZeusdError>>>, Instant, Span);

pub struct QueuePermit(Arc<AtomicUsize>);

impl Drop for QueuePermit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

struct Work<C, R> {
    request: Request<C, R>,
    tracked: Option<(OperationTicket, QueuePermit)>,
}

pub struct DeviceExecutor<C, R> {
    sender: mpsc::Sender<Work<C, R>>,
    tracked: Arc<AtomicUsize>,
}

impl<C, R> Clone for DeviceExecutor<C, R> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            tracked: self.tracked.clone(),
        }
    }
}

impl<C: Send + 'static, R: Send + 'static> DeviceExecutor<C, R> {
    pub fn start<T: Send + 'static>(
        name: String,
        mut device: T,
        execute: fn(&C, &mut T, Instant, bool) -> Result<R, ZeusdError>,
        keepalive: Option<fn(&mut T)>,
    ) -> Result<Self, ZeusdError> {
        let (sender, receiver) = mpsc::channel::<Work<C, R>>();
        std::thread::Builder::new().name(name).spawn(move || {
            let mut next_keepalive = Instant::now() + Duration::from_secs(30);
            loop {
                let work = match keepalive {
                    Some(keepalive) => {
                        if Instant::now() >= next_keepalive {
                            keepalive(&mut device);
                            next_keepalive = Instant::now() + Duration::from_secs(30);
                        }
                        match receiver
                            .recv_timeout(next_keepalive.saturating_duration_since(Instant::now()))
                        {
                            Ok(work) => work,
                            Err(mpsc::RecvTimeoutError::Timeout) => continue,
                            Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        }
                    }
                    None => match receiver.recv() {
                        Ok(work) => work,
                        Err(_) => break,
                    },
                };
                let Work {
                    request: (command, response, accepted, span),
                    mut tracked,
                } = work;
                if let Some((ticket, _)) = &mut tracked {
                    if !ticket.start() {
                        continue;
                    }
                }
                let _guard = span.enter();
                let result = execute(&command, &mut device, accepted, tracked.is_some());
                if let Some((ticket, _)) = &mut tracked {
                    ticket.finish(&result);
                }
                if let Err(error) = &result {
                    tracing::warn!("Device command failed: {error}");
                }
                if let Some(response) = response {
                    if response.blocking_send(result).is_err() {
                        tracing::debug!("Command completed after its caller disconnected");
                    }
                }
            }
        })?;
        Ok(Self {
            sender,
            tracked: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub fn send(&self, request: Request<C, R>) -> Result<(), SendError<Request<C, R>>> {
        self.sender
            .send(Work {
                request,
                tracked: None,
            })
            .map_err(|error| SendError(error.0.request))
    }

    /// Reserve every selected device's queue slot before admitting an operation.
    pub fn reserve(&self) -> Result<QueuePermit, OperationError> {
        self.tracked
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                (count < MAX_TRACKED_QUEUE).then_some(count + 1)
            })
            .map_err(|_| OperationError::capacity("Tracked device queue capacity exhausted"))?;
        Ok(QueuePermit(self.tracked.clone()))
    }

    pub fn send_tracked(&self, command: C, ticket: OperationTicket, permit: QueuePermit) {
        // A closed queue drops its ticket, retaining an unchanged cancellation result.
        let _ = self.sender.send(Work {
            request: (command, None, Instant::now(), Span::current()),
            tracked: Some((ticket, permit)),
        });
    }
}
