//! HTTP and native-executor tests using controllable fake devices.

mod helpers;

use std::net::TcpListener;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use actix_web::{dev::ServerHandle, web, App, HttpServer};
use reqwest::{Client, Method, StatusCode};
use tokio::io::AsyncWriteExt;
use tokio::sync::{mpsc, Notify};
use zeusd::auth::issue_token;
use zeusd::config::ApiGroup;
use zeusd::devices::gpu::{GpuCommand, GpuManagementTasks, GpuManager};
use zeusd::error::ZeusdError;
use zeusd::operations::{
    DeviceState, OperationLimits, OperationRegistry, OperationSnapshot, OutcomeReason,
    SettingEffect,
};
use zeusd::routes::gpu_control_routes;

#[derive(Default)]
struct Gate {
    released: Mutex<bool>,
    changed: Condvar,
    entered: Notify,
}

impl Gate {
    fn block(&self) {
        self.entered.notify_one();
        let mut released = self.released.lock().unwrap();
        while !*released {
            released = self.changed.wait(released).unwrap();
        }
    }

    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.changed.notify_all();
    }
}

struct FakeGpu {
    id: usize,
    gate: Arc<Gate>,
    events: mpsc::UnboundedSender<(usize, u32)>,
}

impl GpuManager for FakeGpu {
    fn device_count() -> Result<u32, ZeusdError> {
        Ok(2)
    }
    fn set_power_management_limit(&mut self, power: u32) -> Result<(), ZeusdError> {
        self.events.send((self.id, power)).unwrap();
        if power == 200_000 {
            self.gate.block();
        }
        if power == 666_000 {
            panic!("simulated native wrapper panic");
        }
        if power == 999_000 {
            return Err(ZeusdError::InvalidRequest(format!(
                "native error: {}",
                "\u{20ac}".repeat(1_000)
            )));
        }
        Ok(())
    }
    fn get_power_management_limit(&mut self) -> Result<u32, ZeusdError> {
        panic!("tracking must not add readback")
    }
    fn get_power_management_limit_constraints(&mut self) -> Result<(u32, u32), ZeusdError> {
        panic!("admission must not read hardware")
    }
    fn set_persistence_mode(&mut self, _: bool) -> Result<(), ZeusdError> {
        Ok(())
    }
    fn get_persistence_mode(&mut self) -> Result<bool, ZeusdError> {
        Ok(true)
    }
    fn set_gpu_locked_clocks(&mut self, _: u32, _: u32) -> Result<(), ZeusdError> {
        Ok(())
    }
    fn reset_gpu_locked_clocks(&mut self) -> Result<(), ZeusdError> {
        Ok(())
    }
    fn set_mem_locked_clocks(&mut self, _: u32, _: u32) -> Result<(), ZeusdError> {
        Ok(())
    }
    fn reset_mem_locked_clocks(&mut self) -> Result<(), ZeusdError> {
        Ok(())
    }
    fn reset_locked_clocks(&mut self) -> Result<(), ZeusdError> {
        Ok(())
    }
    fn get_instant_power_mw(&mut self) -> Result<u32, ZeusdError> {
        Ok(100_000)
    }
    fn get_total_energy_consumption(&mut self) -> Result<u64, ZeusdError> {
        Ok(0)
    }
}

struct Harness {
    base: String,
    client: Client,
    tasks: GpuManagementTasks,
    gates: Vec<Arc<Gate>>,
    events: mpsc::UnboundedReceiver<(usize, u32)>,
    server: ServerHandle,
}

impl Harness {
    async fn start() -> Self {
        Self::with_limits(OperationLimits::default()).await
    }

    async fn with_limits(limits: OperationLimits) -> Self {
        let (events_tx, events) = mpsc::unbounded_channel();
        let gates: Vec<_> = (0..2).map(|_| Arc::new(Gate::default())).collect();
        let tasks = GpuManagementTasks::start(
            gates
                .iter()
                .enumerate()
                .map(|(id, gate)| FakeGpu {
                    id,
                    gate: gate.clone(),
                    events: events_tx.clone(),
                })
                .collect(),
        )
        .unwrap();
        let operations = OperationRegistry::new(limits);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let app_tasks = tasks.clone();
        let app_operations = operations.clone();
        let server = HttpServer::new(move || {
            App::new()
                .app_data(web::Data::new(app_tasks.clone()))
                .app_data(web::Data::new(app_operations.clone()))
                .service(web::scope("/gpu").configure(gpu_control_routes))
        })
        .workers(1)
        .listen(listener)
        .unwrap()
        .run();
        let handle = server.handle();
        tokio::spawn(server);
        Self {
            base: format!("http://127.0.0.1:{port}"),
            client: Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
            tasks,
            gates,
            events,
            server: handle,
        }
    }

    async fn submit(&self, ids: &str, power: u32, deadline: u64) -> OperationSnapshot {
        let response = self.client.post(format!("{}/gpu/set_power_limit/operations?gpu_ids={ids}&power_limit_mw={power}&start_within_ms={deadline}", self.base)).send().await.unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        assert_eq!(response.headers()["cache-control"], "no-store");
        let location = response.headers()["location"].to_str().unwrap().to_owned();
        let snapshot: OperationSnapshot = response.json().await.unwrap();
        assert_eq!(location, snapshot.status_url);
        snapshot
    }

    async fn status(&self, operation: &OperationSnapshot) -> OperationSnapshot {
        let response = self
            .client
            .get(format!("{}{}", self.base, operation.status_url))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["cache-control"], "no-store");
        response.json().await.unwrap()
    }

    async fn wait(
        &self,
        operation: &OperationSnapshot,
        ready: impl Fn(&OperationSnapshot) -> bool,
    ) -> OperationSnapshot {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let status = self.status(operation).await;
                if ready(&status) {
                    return status;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap()
    }

    async fn cancel(&self, operation: &OperationSnapshot) -> OperationSnapshot {
        let response = self
            .client
            .delete(format!("{}{}", self.base, operation.status_url))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["cache-control"], "no-store");
        response.json().await.unwrap()
    }

    async fn block_legacy(&self, id: usize) {
        self.tasks
            .send_command_nonblocking(
                id,
                GpuCommand::SetPowerLimit {
                    power_limit_mw: 200_000,
                },
                Instant::now(),
            )
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), self.gates[id].entered.notified())
            .await
            .unwrap();
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        for gate in &self.gates {
            gate.release();
        }
        let server = self.server.clone();
        tokio::spawn(async move {
            server.stop(false).await;
        });
    }
}

#[tokio::test]
async fn legacy_and_tracked_writes_share_order_and_devices_progress_independently() {
    let mut h = Harness::start().await;
    h.block_legacy(0).await;
    let partial = h.submit("0,1", 250_000, 60_000).await;
    let status = h
        .wait(&partial, |s| s.devices[&1].state == DeviceState::Succeeded)
        .await;
    assert_eq!(status.devices[&0].state, DeviceState::Queued);
    assert!(!status.complete);
    let cancelled = h.cancel(&partial).await;
    assert!(cancelled.complete);
    assert_eq!(cancelled.devices[&0].state, DeviceState::Cancelled);
    assert_eq!(cancelled.devices[&0].effect, SettingEffect::Unchanged);
    assert_eq!(cancelled.devices[&1].state, DeviceState::Succeeded);
    assert_eq!(
        h.cancel(&partial).await.devices[&0].reason,
        Some(OutcomeReason::CancellationRequested)
    );
    let next = h.submit("0", 260_000, 60_000).await;
    h.tasks
        .send_command_nonblocking(
            0,
            GpuCommand::SetPowerLimit {
                power_limit_mw: 270_000,
            },
            Instant::now(),
        )
        .unwrap();
    h.gates[0].release();
    h.wait(&next, |s| s.complete).await;
    let events = tokio::time::timeout(Duration::from_secs(5), async {
        let mut events = Vec::new();
        while events.len() < 4 {
            events.push(h.events.recv().await.unwrap());
        }
        events
    })
    .await
    .unwrap();
    assert_eq!(
        events,
        vec![(0, 200_000), (1, 250_000), (0, 260_000), (0, 270_000)]
    );
}

#[tokio::test]
async fn queue_deadline_expires_behind_blocked_native_call_and_running_cannot_cancel() {
    let mut h = Harness::start().await;
    let running = h.submit("0", 200_000, 60_000).await;
    tokio::time::timeout(Duration::from_secs(5), h.gates[0].entered.notified())
        .await
        .unwrap();
    let queued = h.submit("0", 250_000, 20).await;
    let expired = h.wait(&queued, |s| s.complete).await;
    assert_eq!(
        expired.devices[&0].reason,
        Some(OutcomeReason::DeadlineExceeded)
    );
    assert_eq!(
        h.cancel(&running).await.devices[&0].state,
        DeviceState::Running
    );
    let following = h.submit("0", 260_000, 60_000).await;
    h.gates[0].release();
    assert_eq!(
        h.wait(&running, |s| s.complete).await.devices[&0].state,
        DeviceState::Succeeded
    );
    assert_eq!(
        h.status(&queued).await.devices[&0].state,
        DeviceState::Cancelled
    );
    h.wait(&following, |s| s.complete).await;
    assert_eq!(h.events.try_recv().unwrap(), (0, 200_000));
    assert_eq!(h.events.try_recv().unwrap(), (0, 260_000));
    assert!(h.events.try_recv().is_err());
}

#[tokio::test]
async fn disconnecting_client_does_not_cancel_queued_work() {
    let mut h = Harness::start().await;
    h.block_legacy(0).await;
    assert_eq!(h.events.try_recv().unwrap(), (0, 200_000));
    let mut stream = tokio::net::TcpStream::connect(h.base.strip_prefix("http://").unwrap())
        .await
        .unwrap();
    stream.write_all(b"POST /gpu/set_power_limit/operations?gpu_ids=0,1&power_limit_mw=230000&start_within_ms=60000 HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await.unwrap();
    let started = tokio::time::timeout(Duration::from_secs(5), h.events.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(started, (1, 230_000));
    drop(stream);
    let following = h.submit("0", 250_000, 60_000).await;
    h.gates[0].release();
    assert_eq!(
        h.wait(&following, |s| s.complete).await.devices[&0].state,
        DeviceState::Succeeded
    );
    assert_eq!(h.events.try_recv().unwrap(), (0, 230_000));
    assert_eq!(h.events.try_recv().unwrap(), (0, 250_000));
    assert!(h.events.try_recv().is_err());
}

#[tokio::test]
async fn invalid_admission_never_enqueues_any_selected_device() {
    let mut h = Harness::start().await;
    for query in [
        "gpu_ids=0,2&power_limit_mw=250000&start_within_ms=1000",
        "gpu_ids=0,0&power_limit_mw=250000&start_within_ms=1000",
        "gpu_ids=0&power_limit_mw=0&start_within_ms=1000",
        "gpu_ids=0&power_limit_mw=250000&start_within_ms=0",
        "gpu_ids=0&power_limit_mw=250000",
        "gpu_ids=0&power_limit_mw=250000&start_within_ms=1000&block=false",
        "gpu_ids=0&power_limit_mw=250000&start_within_ms=1000&id=client",
    ] {
        let response = h
            .client
            .post(format!("{}/gpu/set_power_limit/operations?{query}", h.base))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{query}");
        assert_eq!(response.headers()["cache-control"], "no-store");
        let json: serde_json::Value = response.json().await.unwrap();
        assert_eq!(json["error"]["code"], "invalid_request");
    }
    assert!(h.events.try_recv().is_err());
}

#[tokio::test]
async fn executor_panic_retains_uncertainty_and_cancels_unstarted_work() {
    let h = Harness::start().await;
    h.block_legacy(0).await;
    let panic = h.submit("0", 666_000, 60_000).await;
    let queued = h.submit("0", 250_000, 60_000).await;
    h.gates[0].release();
    let failed = h.wait(&panic, |s| s.complete).await;
    assert_eq!(failed.devices[&0].effect, SettingEffect::Uncertain);
    assert_eq!(
        failed.devices[&0].reason,
        Some(OutcomeReason::ExecutorPanicked)
    );
    let cancelled = h.wait(&queued, |s| s.complete).await;
    assert_eq!(cancelled.devices[&0].effect, SettingEffect::Unchanged);
    assert_eq!(
        cancelled.devices[&0].reason,
        Some(OutcomeReason::ExecutorUnavailable)
    );
    let closed = h.submit("0", 260_000, 60_000).await;
    assert_eq!(
        h.wait(&closed, |s| s.complete).await.devices[&0].reason,
        Some(OutcomeReason::ExecutorUnavailable)
    );
}

#[tokio::test]
async fn capacity_rejection_retains_running_work_and_eviction_is_unavailable() {
    let mut h = Harness::with_limits(OperationLimits {
        live: 1,
        completed: 1,
        device_results: 2,
        retention: Duration::from_secs(60),
    })
    .await;
    let running = h.submit("0", 200_000, 60_000).await;
    tokio::time::timeout(Duration::from_secs(5), h.gates[0].entered.notified())
        .await
        .unwrap();
    let response = h.client.post(format!("{}/gpu/set_power_limit/operations?gpu_ids=1&power_limit_mw=250000&start_within_ms=60000", h.base)).send().await.unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let error: serde_json::Value = response.json().await.unwrap();
    assert_eq!(error["error"]["code"], "capacity_exhausted");
    assert_eq!(error["error"]["effect"], "unchanged");
    assert_eq!(h.events.try_recv().unwrap(), (0, 200_000));
    assert!(h.events.try_recv().is_err());
    assert_eq!(
        h.status(&running).await.devices[&0].state,
        DeviceState::Running
    );
    h.gates[0].release();
    h.wait(&running, |s| s.complete).await;
    let next = h.submit("1", 250_000, 60_000).await;
    h.wait(&next, |s| s.complete).await;
    let evicted = h
        .client
        .get(format!("{}{}", h.base, running.status_url))
        .send()
        .await
        .unwrap();
    assert_eq!(evicted.status(), StatusCode::GONE);
    assert_eq!(evicted.headers()["cache-control"], "no-store");
    let error: serde_json::Value = evicted.json().await.unwrap();
    assert_eq!(error["error"]["code"], "operation_unavailable");
    assert_eq!(error["error"]["effect"], "uncertain");
}

#[tokio::test]
async fn backend_rejection_during_execution_is_retained_as_uncertain() {
    let h = Harness::start().await;
    let operation = h.submit("0", 999_000, 60_000).await;
    let failed = h.wait(&operation, |s| s.complete).await;
    assert_eq!(failed.devices[&0].state, DeviceState::Failed);
    assert_eq!(failed.devices[&0].effect, SettingEffect::Uncertain);
    assert_eq!(failed.devices[&0].reason, Some(OutcomeReason::BackendError));
    let message = failed.devices[&0].error.as_ref().unwrap();
    assert!(message.contains("native error"));
    assert!(message.len() < 2100);
    assert_eq!(
        h.cancel(&operation).await.devices[&0].state,
        DeviceState::Failed
    );
}

#[tokio::test]
async fn tracked_routes_require_control_scope_for_every_method() {
    const KEY: &[u8] = b"operations-test-signing-key";
    let app = helpers::TestApp::start_with_auth(KEY).await;
    let client = Client::new();
    let reader = issue_token(
        KEY,
        "reader",
        vec![ApiGroup::GpuRead, ApiGroup::CpuRead],
        None,
    )
    .unwrap();
    let writer = issue_token(
        KEY,
        "writer",
        vec![ApiGroup::GpuControl, ApiGroup::CpuControl],
        None,
    )
    .unwrap();
    for (family, query) in [
        (
            "gpu",
            "gpu_ids=0&power_limit_mw=250000&start_within_ms=60000",
        ),
        (
            "cpu",
            "cpu_ids=0&constraint=long_term&power_limit_mw=150000&start_within_ms=60000",
        ),
    ] {
        let base = format!("http://127.0.0.1:{}", app.port);
        let submit = format!("{base}/{family}/set_power_limit/operations?{query}");
        let response = client
            .post(&submit)
            .bearer_auth(&writer)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let operation: OperationSnapshot = response.json().await.unwrap();
        for (method, url) in [
            (Method::POST, submit),
            (Method::GET, format!("{base}{}", operation.status_url)),
            (Method::DELETE, format!("{base}{}", operation.status_url)),
        ] {
            let forbidden = client
                .request(method.clone(), &url)
                .bearer_auth(&reader)
                .send()
                .await
                .unwrap();
            assert_eq!(forbidden.status(), StatusCode::FORBIDDEN);
            assert_eq!(forbidden.headers()["cache-control"], "no-store");
            let unauthenticated = client.request(method, &url).send().await.unwrap();
            assert_eq!(unauthenticated.status(), StatusCode::UNAUTHORIZED);
        }
        let allowed = client
            .delete(format!("{base}{}", operation.status_url))
            .bearer_auth(&writer)
            .send()
            .await
            .unwrap();
        assert_eq!(allowed.status(), StatusCode::OK);
    }
}

#[tokio::test]
async fn cpu_validation_and_backend_errors_report_different_effects() {
    let app = helpers::TestApp::start_with_cpu_write_errnos(&[Some(5), None]).await;
    let client = Client::new();
    let base = format!("http://127.0.0.1:{}", app.port);
    for (constraint, expected) in [
        ("long_term", SettingEffect::Uncertain),
        ("socket", SettingEffect::Unchanged),
    ] {
        let response = client.post(format!("{base}/cpu/set_power_limit/operations?cpu_ids=0,1&constraint={constraint}&power_limit_mw=150000&start_within_ms=60000")).send().await.unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let operation: OperationSnapshot = response.json().await.unwrap();
        let done = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let s: OperationSnapshot = client
                    .get(format!("{base}{}", operation.status_url))
                    .send()
                    .await
                    .unwrap()
                    .json()
                    .await
                    .unwrap();
                if s.complete {
                    return s;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(done.devices[&0].effect, expected);
        if constraint == "long_term" {
            assert_eq!(done.devices[&1].state, DeviceState::Succeeded);
        }
    }
}
