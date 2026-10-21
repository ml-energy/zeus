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

pub(crate) type Request<C, R> = (C, Option<Sender<Result<R, ZeusdError>>>, Instant, Span);

pub(crate) struct QueuePermit(Arc<AtomicUsize>);

impl Drop for QueuePermit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

struct Work<C, R> {
    request: Request<C, R>,
    tracked: Option<(OperationTicket, QueuePermit)>,
}

pub(crate) struct DeviceExecutor<C, R> {
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
    pub(crate) fn start<T: Send + 'static>(
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

    pub(crate) fn send(&self, request: Request<C, R>) -> Result<(), SendError<Request<C, R>>> {
        self.sender
            .send(Work {
                request,
                tracked: None,
            })
            .map_err(|error| SendError(error.0.request))
    }

    /// Reserve every selected device's queue slot before admitting an operation.
    pub(crate) fn reserve(&self) -> Result<QueuePermit, OperationError> {
        self.tracked
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                (count < MAX_TRACKED_QUEUE).then_some(count + 1)
            })
            .map_err(|_| OperationError::capacity("Tracked device queue capacity exhausted"))?;
        Ok(QueuePermit(self.tracked.clone()))
    }

    pub(crate) fn send_tracked(&self, command: C, ticket: OperationTicket, permit: QueuePermit) {
        // A closed queue drops its ticket, retaining an unchanged cancellation result.
        let _ = self.sender.send(Work {
            request: (command, None, Instant::now(), Span::current()),
            tracked: Some((ticket, permit)),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracked_queue_reservations_are_bounded_and_released() {
        let executor = DeviceExecutor::<(), ()>::start(
            "test-queue-bound".into(),
            (),
            |_, _, _, _| Ok(()),
            None,
        )
        .unwrap();
        let mut permits = (0..MAX_TRACKED_QUEUE)
            .map(|_| executor.reserve().unwrap())
            .collect::<Vec<_>>();
        assert!(executor.reserve().is_err());
        permits.pop();
        assert!(executor.reserve().is_ok());
        drop(permits);
        assert_eq!(executor.tracked.load(Ordering::Relaxed), 0);
    }
}
