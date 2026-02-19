//! Shared power polling and broadcasting infrastructure.
//!
//! Provides generic [`PowerBroadcast`] and [`PowerPoller`] types used by both
//! GPU and CPU power modules. A background task polls device power at a
//! configurable frequency and broadcasts snapshots via a `tokio::sync::watch`
//! channel. Polling is demand-driven: the task sleeps when no subscribers are
//! connected and wakes when the first subscriber arrives.

use std::collections::BTreeSet;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::{watch, Notify};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::WatchStream;
use tokio_stream::Stream;

/// RAII guard that decrements the subscriber count on drop.
pub struct SubscriberGuard {
    count: Arc<AtomicUsize>,
}

impl Drop for SubscriberGuard {
    fn drop(&mut self) {
        self.count.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Shared handle for subscribing to power snapshots.
///
/// Stored as actix-web app data. Cloning creates a new watch subscriber.
#[derive(Clone)]
pub struct PowerBroadcast<T: Clone + Default + Send + Sync + 'static> {
    rx: watch::Receiver<T>,
    subscriber_count: Arc<AtomicUsize>,
    wake: Arc<Notify>,
    /// Set of device indices being monitored.
    valid_ids: Arc<BTreeSet<usize>>,
}

impl<T: Clone + Default + Send + Sync + 'static> PowerBroadcast<T> {
    /// Register a subscriber, waking the poller if it was sleeping.
    ///
    /// Returns an RAII guard that decrements the count on drop.
    pub fn add_subscriber(&self) -> SubscriberGuard {
        self.subscriber_count.fetch_add(1, Ordering::Relaxed);
        self.wake.notify_one();
        SubscriberGuard {
            count: self.subscriber_count.clone(),
        }
    }

    /// Wait for a fresh snapshot from the poller.
    ///
    /// Blocks until the poller broadcasts a new value (ignoring any stale
    /// or default value already in the channel). Returns `None` only if
    /// the poller task has terminated.
    pub async fn wait_for_fresh(&self) -> Option<T> {
        let mut rx = self.rx.clone();
        rx.borrow_and_update();
        if rx.changed().await.is_ok() {
            Some(rx.borrow().clone())
        } else {
            None
        }
    }

    /// Create a stream that yields only fresh snapshots from the poller.
    ///
    /// Skips any stale or default value already in the channel; the first
    /// item is the next value the poller broadcasts after this call.
    pub fn stream(&self) -> impl Stream<Item = T> {
        WatchStream::from_changes(self.rx.clone())
    }

    /// Validate that all requested device indices are being monitored.
    ///
    /// Returns `Ok(())` if all indices are valid, or `Err` with the
    /// set of unknown indices.
    pub fn validate_ids(&self, ids: &[usize]) -> Result<(), Vec<usize>> {
        let unknown: Vec<usize> = ids
            .iter()
            .filter(|id| !self.valid_ids.contains(id))
            .copied()
            .collect();
        if unknown.is_empty() {
            Ok(())
        } else {
            Err(unknown)
        }
    }

    /// Get the set of valid device indices.
    pub fn valid_ids(&self) -> &BTreeSet<usize> {
        &self.valid_ids
    }
}

/// Background task that polls device power at a configured frequency and
/// broadcasts snapshots via a tokio watch channel.
pub struct PowerPoller<T: Clone + Default + Send + Sync + 'static> {
    broadcast: PowerBroadcast<T>,
    _handle: JoinHandle<()>,
}

impl<T: Clone + Default + Send + Sync + 'static> PowerPoller<T> {
    /// Start the power polling background task.
    ///
    /// `spawn_task` receives the watch sender, subscriber count, and wake
    /// notifier, and should return a future that polls devices and sends
    /// snapshots on the watch channel.
    pub fn start<F, Fut>(valid_ids: BTreeSet<usize>, spawn_task: F) -> Self
    where
        F: FnOnce(watch::Sender<T>, Arc<AtomicUsize>, Arc<Notify>) -> Fut,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let (tx, rx) = watch::channel(T::default());
        let subscriber_count = Arc::new(AtomicUsize::new(0));
        let wake = Arc::new(Notify::new());
        let handle = tokio::spawn(spawn_task(tx, subscriber_count.clone(), wake.clone()));
        Self {
            broadcast: PowerBroadcast {
                rx,
                subscriber_count,
                wake,
                valid_ids: Arc::new(valid_ids),
            },
            _handle: handle,
        }
    }

    /// Get the broadcast handle for sharing with route handlers.
    pub fn broadcast(&self) -> PowerBroadcast<T> {
        self.broadcast.clone()
    }
}
