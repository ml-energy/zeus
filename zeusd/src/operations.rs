//! Retained results and cancellation for asynchronous power limit commands.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use actix_web::{http::StatusCode, HttpResponse, ResponseError};
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use uuid::Uuid;

use crate::error::ZeusdError;

/// Bounds shared by GPU and CPU operations in one server.
#[derive(Clone, Copy, Debug)]
pub struct OperationLimits {
    pub live: usize,
    pub completed: usize,
    pub device_results: usize,
    pub retention: Duration,
}

impl Default for OperationLimits {
    fn default() -> Self {
        Self {
            live: 1024,
            completed: 4096,
            device_results: 16384,
            retention: Duration::from_secs(3600),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeviceType {
    Gpu,
    Cpu,
}

impl DeviceType {
    fn path(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
        }
    }
}

/// Evidence about this command's effect, independent of other writers.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SettingEffect {
    Unchanged,
    Applied,
    Restored,
    Uncertain,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeviceState {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl DeviceState {
    fn terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeReason {
    CancellationRequested,
    DeadlineExceeded,
    ExecutorUnavailable,
    ExecutorPanicked,
    BackendError,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DeviceOutcome {
    pub state: DeviceState,
    pub effect: SettingEffect,
    pub reason: Option<OutcomeReason>,
    pub error: Option<String>,
}

impl DeviceOutcome {
    fn queued() -> Self {
        Self {
            state: DeviceState::Queued,
            effect: SettingEffect::Unchanged,
            reason: None,
            error: None,
        }
    }

    fn cancel(&mut self, reason: OutcomeReason) {
        if self.state == DeviceState::Queued {
            self.state = DeviceState::Cancelled;
            self.reason = Some(reason);
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RequestedSetting {
    pub power_limit_mw: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constraint: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OperationSnapshot {
    pub id: String,
    pub status_url: String,
    pub device_type: DeviceType,
    pub requested: RequestedSetting,
    pub start_within_ms: u64,
    pub complete: bool,
    pub devices: BTreeMap<usize, DeviceOutcome>,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OperationErrorCode {
    InvalidRequest,
    CapacityExhausted,
    OperationUnavailable,
    OperationNotFound,
}

#[derive(Debug, thiserror::Error, Serialize)]
#[error("{message}")]
pub struct OperationError {
    pub code: OperationErrorCode,
    pub message: String,
    pub effect: SettingEffect,
}

impl OperationError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            code: OperationErrorCode::InvalidRequest,
            message: message.into(),
            effect: SettingEffect::Unchanged,
        }
    }

    pub fn capacity(message: impl Into<String>) -> Self {
        Self {
            code: OperationErrorCode::CapacityExhausted,
            message: message.into(),
            effect: SettingEffect::Unchanged,
        }
    }

    fn unavailable(message: &str) -> Self {
        Self {
            code: OperationErrorCode::OperationUnavailable,
            message: message.into(),
            effect: SettingEffect::Uncertain,
        }
    }

    fn not_found() -> Self {
        Self {
            code: OperationErrorCode::OperationNotFound,
            message:
                "No operation with this ID was admitted for this device type in this server startup"
                    .into(),
            effect: SettingEffect::Unchanged,
        }
    }
}

impl ResponseError for OperationError {
    fn status_code(&self) -> StatusCode {
        match self.code {
            OperationErrorCode::InvalidRequest => StatusCode::BAD_REQUEST,
            OperationErrorCode::CapacityExhausted => StatusCode::SERVICE_UNAVAILABLE,
            OperationErrorCode::OperationUnavailable => StatusCode::GONE,
            OperationErrorCode::OperationNotFound => StatusCode::NOT_FOUND,
        }
    }

    fn error_response(&self) -> HttpResponse {
        HttpResponse::build(self.status_code())
            .insert_header(("Cache-Control", "no-store"))
            .json(serde_json::json!({ "error": self }))
    }
}

struct Record {
    snapshot: OperationSnapshot,
    deadline: Instant,
}

#[derive(Default)]
struct State {
    counter: u64,
    records: HashMap<u64, Record>,
    deadlines: BTreeSet<(Instant, u64)>,
    completed: VecDeque<(Instant, u64)>,
    live: usize,
    device_results: usize,
}

struct Inner {
    startup: Uuid,
    limits: OperationLimits,
    state: Mutex<State>,
    wake: Arc<Notify>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.wake.notify_one();
    }
}

/// One registry is shared by all HTTP workers and both device families.
#[derive(Clone)]
pub struct OperationRegistry(Arc<Inner>);

impl Default for OperationRegistry {
    fn default() -> Self {
        Self::new(OperationLimits::default())
    }
}

impl OperationRegistry {
    pub fn new(limits: OperationLimits) -> Self {
        assert!(limits.live > 0 && limits.completed > 0 && limits.device_results > 0);
        assert!(!limits.retention.is_zero());
        let inner = Arc::new(Inner {
            startup: Uuid::new_v4(),
            limits,
            state: Mutex::new(State::default()),
            wake: Arc::new(Notify::new()),
        });
        tokio::spawn(maintain(Arc::downgrade(&inner), inner.wake.clone()));
        Self(inner)
    }

    /// Reserve retained results before any command can be enqueued.
    pub fn admit(
        &self,
        device_type: DeviceType,
        ids: &[usize],
        requested: RequestedSetting,
        start_within_ms: u64,
    ) -> Result<(OperationSnapshot, Vec<OperationTicket>), OperationError> {
        if start_within_ms == 0 {
            return Err(OperationError::invalid("start_within_ms must be positive"));
        }
        let now = Instant::now();
        let deadline = now
            .checked_add(Duration::from_millis(start_within_ms))
            .ok_or_else(|| OperationError::invalid("start_within_ms is out of range"))?;
        if ids.is_empty() || ids.iter().copied().collect::<BTreeSet<_>>().len() != ids.len() {
            return Err(OperationError::invalid(
                "Device IDs must be nonempty and unique",
            ));
        }
        let mut state = self.0.state.lock().unwrap();
        state.maintain(now, self.0.limits);
        if state.live >= self.0.limits.live || ids.len() > self.0.limits.device_results {
            return Err(OperationError::capacity(
                "Tracked operation capacity exhausted",
            ));
        }
        while state.device_results + ids.len() > self.0.limits.device_results {
            if !state.evict_completed() {
                return Err(OperationError::capacity(
                    "Retained device result capacity exhausted",
                ));
            }
        }
        let counter = state
            .counter
            .checked_add(1)
            .ok_or_else(|| OperationError::capacity("Operation counter exhausted"))?;
        state.counter = counter;
        let id = format!("{}:{counter}", self.0.startup);
        let snapshot = OperationSnapshot {
            status_url: format!("/{}/set_power_limit/operations/{id}", device_type.path()),
            id,
            device_type,
            requested,
            start_within_ms,
            complete: false,
            devices: ids
                .iter()
                .map(|&id| (id, DeviceOutcome::queued()))
                .collect(),
        };
        state.records.insert(
            counter,
            Record {
                snapshot: snapshot.clone(),
                deadline,
            },
        );
        state.deadlines.insert((deadline, counter));
        state.live += 1;
        state.device_results += ids.len();
        let tickets = ids
            .iter()
            .map(|&device| OperationTicket {
                inner: Arc::downgrade(&self.0),
                counter,
                device,
                settled: false,
            })
            .collect();
        self.0.wake.notify_one();
        Ok((snapshot, tickets))
    }

    pub fn get(
        &self,
        id: &str,
        device_type: DeviceType,
    ) -> Result<OperationSnapshot, OperationError> {
        self.access(id, device_type, false)
    }

    pub fn cancel(
        &self,
        id: &str,
        device_type: DeviceType,
    ) -> Result<OperationSnapshot, OperationError> {
        self.access(id, device_type, true)
    }

    fn access(
        &self,
        id: &str,
        device_type: DeviceType,
        cancel: bool,
    ) -> Result<OperationSnapshot, OperationError> {
        let (startup, counter) = id
            .rsplit_once(':')
            .ok_or_else(|| OperationError::invalid("Malformed operation ID"))?;
        let startup = Uuid::parse_str(startup)
            .map_err(|_| OperationError::invalid("Malformed operation startup UUID"))?;
        let counter = counter
            .parse::<u64>()
            .ok()
            .filter(|&counter| counter > 0)
            .ok_or_else(|| OperationError::invalid("Malformed operation counter"))?;
        if startup != self.0.startup {
            return Err(OperationError::unavailable(
                "This server startup has no authoritative result for this operation",
            ));
        }
        let now = Instant::now();
        let mut state = self.0.state.lock().unwrap();
        state.maintain(now, self.0.limits);
        let last_counter = state.counter;
        let record = match state.records.get_mut(&counter) {
            Some(record) if record.snapshot.device_type == device_type => record,
            Some(_) => return Err(OperationError::not_found()),
            None if counter <= last_counter => {
                return Err(OperationError::unavailable(
                    "The result is no longer retained; the operation may have executed",
                ))
            }
            None => return Err(OperationError::not_found()),
        };
        if cancel {
            for outcome in record.snapshot.devices.values_mut() {
                outcome.cancel(OutcomeReason::CancellationRequested);
            }
        }
        state.complete(counter, now);
        // Copy before enforcing retention: a cancellation response remains available
        // even if this completion displaces another retained result.
        let snapshot = state.records[&counter].snapshot.clone();
        state.maintain(now, self.0.limits);
        self.0.wake.notify_one();
        Ok(snapshot)
    }
}

impl State {
    fn complete(&mut self, counter: u64, now: Instant) {
        let record = self.records.get_mut(&counter).unwrap();
        if !record.snapshot.complete && record.snapshot.devices.values().all(|d| d.state.terminal())
        {
            record.snapshot.complete = true;
            self.live -= 1;
            self.deadlines.remove(&(record.deadline, counter));
            self.completed.push_back((now, counter));
        }
    }

    fn evict_completed(&mut self) -> bool {
        match self.completed.pop_front() {
            Some((_, counter)) => {
                let record = self.records.remove(&counter).unwrap();
                self.device_results -= record.snapshot.devices.len();
                true
            }
            None => false,
        }
    }

    fn maintain(&mut self, now: Instant, limits: OperationLimits) {
        while let Some(&(deadline, counter)) = self.deadlines.first() {
            if deadline > now {
                break;
            }
            self.deadlines.pop_first();
            let record = self.records.get_mut(&counter).unwrap();
            for outcome in record.snapshot.devices.values_mut() {
                outcome.cancel(OutcomeReason::DeadlineExceeded);
            }
            self.complete(counter, now);
        }
        while self.completed.front().is_some_and(|&(completed, _)| {
            self.completed.len() > limits.completed
                || now.duration_since(completed) >= limits.retention
        }) {
            self.evict_completed();
        }
    }

    fn next_wake(&self, retention: Duration) -> Option<Instant> {
        let deadline = self.deadlines.first().map(|&(deadline, _)| deadline);
        let expiry = self
            .completed
            .front()
            .and_then(|&(completed, _)| completed.checked_add(retention));
        deadline.into_iter().chain(expiry).min()
    }
}

async fn maintain(inner: Weak<Inner>, wake: Arc<Notify>) {
    loop {
        let next = match inner.upgrade() {
            Some(inner) => {
                let mut state = inner.state.lock().unwrap();
                state.maintain(Instant::now(), inner.limits);
                state.next_wake(inner.limits.retention)
            }
            None => return,
        };
        match next {
            Some(next) => tokio::select! {
                _ = tokio::time::sleep_until(next.into()) => {},
                _ = wake.notified() => {},
            },
            None => wake.notified().await,
        }
    }
}

/// A queue entry holds only a weak reference, so cancelled records can expire.
pub struct OperationTicket {
    inner: Weak<Inner>,
    counter: u64,
    device: usize,
    settled: bool,
}

impl OperationTicket {
    /// Competes with cancellation while holding the registry's state lock.
    pub(crate) fn start(&mut self) -> bool {
        let Some(inner) = self.inner.upgrade() else {
            return false;
        };
        let now = Instant::now();
        let mut state = inner.state.lock().unwrap();
        state.maintain(now, inner.limits);
        let Some(record) = state.records.get_mut(&self.counter) else {
            return false;
        };
        let outcome = record.snapshot.devices.get_mut(&self.device).unwrap();
        if outcome.state != DeviceState::Queued {
            return false;
        }
        outcome.state = DeviceState::Running;
        outcome.effect = SettingEffect::Uncertain;
        true
    }

    pub(crate) fn finish<T>(&mut self, result: &Result<T, ZeusdError>) {
        self.update(|outcome| {
            match result {
                Ok(_) => {
                    outcome.state = DeviceState::Succeeded;
                    outcome.effect = SettingEffect::Applied;
                }
                Err(error) => {
                    outcome.state = DeviceState::Failed;
                    outcome.effect = error.setting_effect();
                    outcome.reason = Some(OutcomeReason::BackendError);
                    let mut message = error.to_string();
                    // Bound vendor and external-command messages in retained records.
                    if message.len() > 2048 {
                        let mut end = 2048;
                        while !message.is_char_boundary(end) {
                            end -= 1;
                        }
                        let marker = " [truncated]";
                        let mut bounded = String::with_capacity(end + marker.len());
                        bounded.push_str(&message[..end]);
                        bounded.push_str(marker);
                        message = bounded;
                    }
                    outcome.error = Some(message);
                }
            }
        });
        self.settled = true;
    }

    fn update(&self, update: impl FnOnce(&mut DeviceOutcome)) {
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        let mut state = inner.state.lock().unwrap();
        let Some(record) = state.records.get_mut(&self.counter) else {
            return;
        };
        let outcome = record.snapshot.devices.get_mut(&self.device).unwrap();
        if outcome.state.terminal() {
            return;
        }
        update(outcome);
        let now = Instant::now();
        state.complete(self.counter, now);
        state.maintain(now, inner.limits);
        inner.wake.notify_one();
    }
}

impl Drop for OperationTicket {
    fn drop(&mut self) {
        if !self.settled {
            self.update(|outcome| match outcome.state {
                DeviceState::Queued => outcome.cancel(OutcomeReason::ExecutorUnavailable),
                DeviceState::Running => {
                    outcome.state = DeviceState::Failed;
                    outcome.effect = SettingEffect::Uncertain;
                    outcome.reason = Some(OutcomeReason::ExecutorPanicked);
                }
                _ => {}
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn admit(
        registry: &OperationRegistry,
        ids: &[usize],
        deadline: u64,
    ) -> (OperationSnapshot, Vec<OperationTicket>) {
        registry
            .admit(
                DeviceType::Gpu,
                ids,
                RequestedSetting {
                    power_limit_mw: 200_000,
                    constraint: None,
                },
                deadline,
            )
            .unwrap()
    }

    #[tokio::test]
    async fn cancellation_and_execution_compete_for_one_owner() {
        let registry = OperationRegistry::default();
        for _ in 0..128 {
            let (snapshot, mut tickets) = admit(&registry, &[0], 60_000);
            let mut ticket = tickets.pop().unwrap();
            let barrier = Arc::new(std::sync::Barrier::new(2));
            let worker_barrier = barrier.clone();
            let worker = std::thread::spawn(move || {
                worker_barrier.wait();
                let started = ticket.start();
                (started, ticket)
            });
            barrier.wait();
            let cancelled = registry.cancel(&snapshot.id, DeviceType::Gpu).unwrap();
            let (started, mut ticket) = worker.join().unwrap();
            if started {
                assert_eq!(cancelled.devices[&0].state, DeviceState::Running);
                ticket.finish(&Ok::<(), ZeusdError>(()));
                assert_eq!(
                    registry.get(&snapshot.id, DeviceType::Gpu).unwrap().devices[&0].state,
                    DeviceState::Succeeded
                );
            } else {
                assert_eq!(cancelled.devices[&0].state, DeviceState::Cancelled);
            }
        }
    }

    #[tokio::test]
    async fn deadline_timer_expires_without_status_requests_or_queue_progress() {
        let registry = OperationRegistry::default();
        let (snapshot, mut tickets) = admit(&registry, &[0, 1], 20);
        assert!(tickets[0].start());
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let expired = {
                    let state = registry.0.state.lock().unwrap();
                    let record = state.records.values().next().unwrap();
                    record.snapshot.devices[&1].state == DeviceState::Cancelled
                };
                if expired {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        let status = registry.get(&snapshot.id, DeviceType::Gpu).unwrap();
        assert_eq!(status.devices[&0].state, DeviceState::Running);
        assert_eq!(
            status.devices[&1].reason,
            Some(OutcomeReason::DeadlineExceeded)
        );
        assert!(!status.complete);
        assert!(!tickets[1].start());
        tickets[0].finish(&Ok::<(), ZeusdError>(()));
        assert!(
            registry
                .get(&snapshot.id, DeviceType::Gpu)
                .unwrap()
                .complete
        );
    }

    #[tokio::test]
    async fn bounds_evict_only_completed_results_and_distinguish_unavailable_ids() {
        let registry = OperationRegistry::new(OperationLimits {
            live: 2,
            completed: 1,
            device_results: 3,
            retention: Duration::from_secs(60),
        });
        let (first, mut first_tickets) = admit(&registry, &[0, 1], 60_000);
        assert!(first_tickets[0].start());
        let (second, second_tickets) = admit(&registry, &[2], 60_000);
        let request = || registry.admit(DeviceType::Gpu, &[3], first.requested.clone(), 60_000);
        assert_eq!(
            request().err().unwrap().code,
            OperationErrorCode::CapacityExhausted
        );
        registry.cancel(&second.id, DeviceType::Gpu).unwrap();
        let (third, _third_tickets) = request().unwrap();
        assert_eq!(
            registry.get(&second.id, DeviceType::Gpu).unwrap_err().code,
            OperationErrorCode::OperationUnavailable
        );
        assert_eq!(
            registry.get(&first.id, DeviceType::Gpu).unwrap().devices[&0].state,
            DeviceState::Running
        );
        registry.cancel(&third.id, DeviceType::Gpu).unwrap();
        let fourth = request().unwrap().0;
        registry.cancel(&fourth.id, DeviceType::Gpu).unwrap();
        assert!(registry.get(&third.id, DeviceType::Gpu).is_err());
        drop(second_tickets);
        assert_eq!(
            registry
                .get(&format!("{}:9999", registry.0.startup), DeviceType::Gpu)
                .unwrap_err()
                .code,
            OperationErrorCode::OperationNotFound
        );
        assert_eq!(
            registry
                .get(&format!("{}:1", Uuid::new_v4()), DeviceType::Gpu)
                .unwrap_err()
                .code,
            OperationErrorCode::OperationUnavailable
        );
        assert_eq!(
            registry.get(&first.id, DeviceType::Cpu).unwrap_err().code,
            OperationErrorCode::OperationNotFound
        );
        let state = registry.0.state.lock().unwrap();
        assert!(state.device_results <= 3);
        assert_eq!(state.live, 1);
    }

    #[tokio::test]
    async fn retention_expiry_and_counter_exhaustion_do_not_create_work() {
        let registry = OperationRegistry::new(OperationLimits {
            retention: Duration::from_millis(20),
            ..OperationLimits::default()
        });
        let (snapshot, tickets) = admit(&registry, &[0], 60_000);
        registry.cancel(&snapshot.id, DeviceType::Gpu).unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while !registry.0.state.lock().unwrap().records.is_empty() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        drop(tickets);
        assert_eq!(
            registry
                .get(&snapshot.id, DeviceType::Gpu)
                .unwrap_err()
                .code,
            OperationErrorCode::OperationUnavailable
        );
        registry.0.state.lock().unwrap().counter = u64::MAX;
        assert!(registry
            .admit(DeviceType::Gpu, &[0], snapshot.requested, 1)
            .is_err());
        assert!(registry.0.state.lock().unwrap().records.is_empty());
    }

    #[tokio::test]
    async fn failed_native_calls_are_uncertain_without_explicit_backend_evidence() {
        let registry = OperationRegistry::default();
        let (snapshot, mut tickets) = admit(&registry, &[0, 1, 2], 60_000);
        for ticket in &mut tickets {
            assert!(ticket.start());
        }
        tickets[0].finish(&Err::<(), _>(ZeusdError::InvalidRequest(
            "backend rejected after writing".into(),
        )));
        tickets[1].finish(&Err::<(), _>(
            ZeusdError::InvalidRequest("validation failed".into())
                .with_setting_effect(SettingEffect::Unchanged),
        ));
        tickets[2].finish(&Err::<(), _>(ZeusdError::InvalidRequest(
            "x".repeat(10_000),
        )));
        let status = registry.get(&snapshot.id, DeviceType::Gpu).unwrap();
        assert!(status.complete);
        assert_eq!(status.devices[&0].effect, SettingEffect::Uncertain);
        assert_eq!(status.devices[&1].effect, SettingEffect::Unchanged);
        assert!(status.devices[&2].error.as_ref().unwrap().len() < 2100);
        let state = registry.0.state.lock().unwrap();
        let retained = &state.records.values().next().unwrap().snapshot.devices[&2];
        assert!(retained.error.as_ref().unwrap().capacity() < 4096);
    }
}
