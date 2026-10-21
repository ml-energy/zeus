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
    counter: u64,
    device_type: DeviceType,
    device_results: usize,
    deadline: Instant,
    snapshot: Mutex<OperationSnapshot>,
}

impl Record {
    fn expire(&self, snapshot: &mut OperationSnapshot, now: Instant) {
        if now >= self.deadline {
            for outcome in snapshot.devices.values_mut() {
                outcome.cancel(OutcomeReason::DeadlineExceeded);
            }
        }
    }

    fn complete(&self, snapshot: &mut OperationSnapshot, inner: &Inner) {
        if !snapshot.complete && snapshot.devices.values().all(|d| d.state.terminal()) {
            snapshot.complete = true;
            let retired = {
                let mut state = inner.state.lock().unwrap();
                let now = Instant::now();
                state.live -= 1;
                state.deadlines.remove(&(self.deadline, self.counter));
                state.completed.push_back((now, self.counter));
                state.maintain_completed(now, inner.limits)
            };
            drop(retired);
            inner.wake.notify_one();
        }
    }
}

#[derive(Default)]
struct State {
    counter: u64,
    records: HashMap<u64, Arc<Record>>,
    deadlines: BTreeSet<(Instant, u64)>,
    completed: VecDeque<(Instant, u64)>,
    live: usize,
    device_results: usize,
}

struct Inner {
    startup: Uuid,
    limits: OperationLimits,
    // Lock an operation before the registry when both locks are needed.
    // Registry code never acquires an operation lock while holding this lock.
    state: Mutex<State>,
    wake: Arc<Notify>,
}

impl Inner {
    fn expired_records(&self) -> Vec<Arc<Record>> {
        let (expired, retired) = {
            let mut state = self.state.lock().unwrap();
            let now = Instant::now();
            let expired = state
                .deadlines
                .range(..=(now, u64::MAX))
                .map(|&(_, counter)| state.records[&counter].clone())
                .collect();
            (expired, state.maintain_completed(now, self.limits))
        };
        drop(retired);
        expired
    }

    fn maintain(&self) {
        for record in self.expired_records() {
            let mut snapshot = record.snapshot.lock().unwrap();
            record.expire(&mut snapshot, Instant::now());
            record.complete(&mut snapshot, self);
            self.state
                .lock()
                .unwrap()
                .deadlines
                .remove(&(record.deadline, record.counter));
        }
    }
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
        let devices: BTreeMap<_, _> = ids
            .iter()
            .map(|&id| (id, DeviceOutcome::queued()))
            .collect();
        let retained_devices = devices.clone();
        let retained_requested = requested.clone();
        self.0.maintain();
        let mut retired = Vec::new();
        let mut state = self.0.state.lock().unwrap();
        if state.live >= self.0.limits.live || ids.len() > self.0.limits.device_results {
            return Err(OperationError::capacity(
                "Tracked operation capacity exhausted",
            ));
        }
        let remaining_capacity = self.0.limits.device_results - ids.len();
        if state.device_results > remaining_capacity {
            let reclaimable: usize = state
                .completed
                .iter()
                .map(|&(_, counter)| state.records[&counter].device_results)
                .sum();
            if state.device_results - reclaimable > remaining_capacity {
                return Err(OperationError::capacity(
                    "Retained device result capacity exhausted",
                ));
            }
        }
        let counter = state
            .counter
            .checked_add(1)
            .ok_or_else(|| OperationError::capacity("Operation counter exhausted"))?;
        while state.device_results > remaining_capacity {
            retired.push(state.evict_completed().unwrap());
        }
        state.counter = counter;
        let id = format!("{}:{counter}", self.0.startup);
        let snapshot = OperationSnapshot {
            status_url: format!("/{}/set_power_limit/operations/{id}", device_type.path()),
            id,
            device_type,
            requested,
            start_within_ms,
            complete: false,
            devices,
        };
        let record = Arc::new(Record {
            counter,
            device_type,
            device_results: ids.len(),
            deadline,
            snapshot: Mutex::new(OperationSnapshot {
                id: snapshot.id.clone(),
                status_url: snapshot.status_url.clone(),
                requested: retained_requested,
                devices: retained_devices,
                ..snapshot
            }),
        });
        state.records.insert(counter, record.clone());
        state.deadlines.insert((deadline, counter));
        state.live += 1;
        state.device_results += ids.len();
        drop(state);
        drop(retired);
        let tickets = ids
            .iter()
            .map(|&device| OperationTicket {
                inner: Arc::downgrade(&self.0),
                record: Arc::downgrade(&record),
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
        let mut retired = Vec::new();
        let record = {
            let mut state = self.0.state.lock().unwrap();
            retired.extend(state.maintain_completed(Instant::now(), self.0.limits));
            match state.records.get(&counter) {
                Some(record) if record.device_type == device_type => record.clone(),
                Some(_) => return Err(OperationError::not_found()),
                None if counter <= state.counter => {
                    return Err(OperationError::unavailable(
                        "The result is no longer retained; the operation may have executed",
                    ))
                }
                None => return Err(OperationError::not_found()),
            }
        };
        drop(retired);
        let mut snapshot = record.snapshot.lock().unwrap();
        record.expire(&mut snapshot, Instant::now());
        if cancel {
            for outcome in snapshot.devices.values_mut() {
                outcome.cancel(OutcomeReason::CancellationRequested);
            }
        }
        record.complete(&mut snapshot, &self.0);
        Ok(snapshot.clone())
    }
}

impl State {
    fn evict_completed(&mut self) -> Option<Arc<Record>> {
        let (_, counter) = self.completed.pop_front()?;
        let record = self.records.remove(&counter).unwrap();
        self.device_results -= record.device_results;
        Some(record)
    }

    fn maintain_completed(&mut self, now: Instant, limits: OperationLimits) -> Vec<Arc<Record>> {
        let mut retired = Vec::new();
        while self.completed.front().is_some_and(|&(completed, _)| {
            self.completed.len() > limits.completed
                || now.duration_since(completed) >= limits.retention
        }) {
            retired.push(self.evict_completed().unwrap());
        }
        retired
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
                inner.maintain();
                let state = inner.state.lock().unwrap();
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

/// A queue entry holds only weak references, so cancelled records can expire.
pub struct OperationTicket {
    inner: Weak<Inner>,
    record: Weak<Record>,
    device: usize,
    settled: bool,
}

impl OperationTicket {
    /// Competes with cancellation while holding the operation's lock.
    pub fn start(&mut self) -> bool {
        let Some(inner) = self.inner.upgrade() else {
            return false;
        };
        let Some(record) = self.record.upgrade() else {
            return false;
        };
        let mut snapshot = record.snapshot.lock().unwrap();
        record.expire(&mut snapshot, Instant::now());
        record.complete(&mut snapshot, &inner);
        let outcome = snapshot.devices.get_mut(&self.device).unwrap();
        if outcome.state != DeviceState::Queued {
            return false;
        }
        outcome.state = DeviceState::Running;
        outcome.effect = SettingEffect::Uncertain;
        true
    }

    pub fn finish<T>(&mut self, result: &Result<T, ZeusdError>) {
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
        let Some(record) = self.record.upgrade() else {
            return;
        };
        let mut snapshot = record.snapshot.lock().unwrap();
        let outcome = snapshot.devices.get_mut(&self.device).unwrap();
        if outcome.state.terminal() {
            return;
        }
        update(outcome);
        record.expire(&mut snapshot, Instant::now());
        record.complete(&mut snapshot, &inner);
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
    async fn insufficient_capacity_preserves_completed_results() {
        let registry = OperationRegistry::new(OperationLimits {
            live: 2,
            completed: 1,
            device_results: 3,
            retention: Duration::from_secs(60),
        });
        let (active, _active_tickets) = admit(&registry, &[0, 1], 60_000);
        let (completed, mut completed_tickets) = admit(&registry, &[2], 60_000);
        registry.cancel(&completed.id, DeviceType::Gpu).unwrap();

        let error = registry
            .admit(DeviceType::Gpu, &[3, 4], active.requested, 60_000)
            .err()
            .unwrap();
        assert_eq!(error.code, OperationErrorCode::CapacityExhausted);
        assert!(
            registry
                .get(&completed.id, DeviceType::Gpu)
                .unwrap()
                .complete
        );
        let (_replacement, _tickets) = admit(&registry, &[3], 60_000);
        assert_eq!(
            registry
                .get(&completed.id, DeviceType::Gpu)
                .unwrap_err()
                .code,
            OperationErrorCode::OperationUnavailable
        );
        assert!(!completed_tickets[0].start());
        assert!(!registry.get(&active.id, DeviceType::Gpu).unwrap().complete);
    }

    #[tokio::test]
    async fn completed_results_expire_with_outstanding_tickets() {
        let registry = OperationRegistry::new(OperationLimits {
            retention: Duration::from_millis(20),
            ..OperationLimits::default()
        });
        let (snapshot, mut tickets) = admit(&registry, &[0], 60_000);
        registry.cancel(&snapshot.id, DeviceType::Gpu).unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(
            registry
                .get(&snapshot.id, DeviceType::Gpu)
                .unwrap_err()
                .code,
            OperationErrorCode::OperationUnavailable
        );
        assert!(!tickets[0].start());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_completion_and_eviction_preserve_capacity() {
        let registry = OperationRegistry::new(OperationLimits {
            live: 4,
            completed: 2,
            device_results: 8,
            retention: Duration::from_secs(60),
        });
        let operations = (0..4)
            .map(|_| admit(&registry, &[0, 1], 60_000))
            .collect::<Vec<_>>();
        let ids: Vec<_> = operations.iter().map(|(s, _)| s.id.clone()).collect();
        let barrier = Arc::new(std::sync::Barrier::new(12));
        std::thread::scope(|scope| {
            for (operation, tickets) in operations {
                for mut ticket in tickets {
                    let barrier = barrier.clone();
                    scope.spawn(move || {
                        barrier.wait();
                        if ticket.start() {
                            ticket.finish(&Ok::<(), ZeusdError>(()));
                        }
                    });
                }
                let barrier = barrier.clone();
                let registry = registry.clone();
                scope.spawn(move || {
                    barrier.wait();
                    if let Err(error) = registry.cancel(&operation.id, DeviceType::Gpu) {
                        assert_eq!(error.code, OperationErrorCode::OperationUnavailable);
                    }
                });
            }
        });
        let mut retained = 0;
        for id in ids {
            match registry.get(&id, DeviceType::Gpu) {
                Ok(snapshot) => {
                    assert!(snapshot.complete);
                    retained += 1;
                }
                Err(error) => assert_eq!(error.code, OperationErrorCode::OperationUnavailable),
            }
        }
        assert_eq!(retained, 2);
        let next: Vec<_> = (0..4).map(|_| admit(&registry, &[0, 1], 60_000)).collect();
        let error = registry
            .admit(DeviceType::Gpu, &[2], next[0].0.requested.clone(), 60_000)
            .err()
            .unwrap();
        assert_eq!(error.code, OperationErrorCode::CapacityExhausted);
    }

    #[tokio::test]
    async fn expired_queued_work_releases_admission_capacity() {
        let registry = OperationRegistry::new(OperationLimits {
            live: 1,
            completed: 1,
            device_results: 1,
            retention: Duration::from_secs(60),
        });
        let (first, mut tickets) = admit(&registry, &[0], 20);
        tokio::time::sleep(Duration::from_millis(20)).await;
        let (second, _tickets) = admit(&registry, &[1], 60_000);
        assert!(!registry.get(&second.id, DeviceType::Gpu).unwrap().complete);
        assert_eq!(
            registry.get(&first.id, DeviceType::Gpu).unwrap_err().code,
            OperationErrorCode::OperationUnavailable
        );
        assert!(!tickets[0].start());
    }
}
