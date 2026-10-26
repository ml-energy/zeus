//! Command lifecycles through HTTP, device queues, and controllable native writes.

mod helpers;

use std::net::TcpListener;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use actix_web::{dev::ServerHandle, web, App, HttpServer};
use reqwest::{Client, Method, RequestBuilder, StatusCode};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use zeusd::auth::issue_token;
use zeusd::config::ApiGroup;
use zeusd::devices::gpu::{GpuManagementTasks, GpuManager};
use zeusd::error::ZeusdError;
use zeusd::operations::{
    DeviceState, OperationLimits, OperationRegistry, OperationSnapshot, OutcomeReason,
    SettingEffect,
};
use zeusd::routes::gpu_control_routes;

const OPERATIONS: &str = "/gpu/set_power_limit/operations";
const TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Default)]
struct Gate {
    released: Mutex<bool>,
    changed: Condvar,
}

impl Gate {
    fn block(&self) {
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
    limit: u32,
    gate: Arc<Gate>,
    calls: Arc<Mutex<Vec<(usize, u32)>>>,
}

impl GpuManager for FakeGpu {
    fn device_count() -> Result<u32, ZeusdError> {
        Ok(2)
    }
    fn set_power_management_limit(&mut self, power: u32) -> Result<(), ZeusdError> {
        self.calls.lock().unwrap().push((self.id, power));
        match power {
            200_000 => self.gate.block(),
            666_000 => panic!("simulated native wrapper panic"),
            999_000 => {
                return Err(ZeusdError::InvalidRequest(
                    "simulated native failure".into(),
                ))
            }
            _ => {}
        }
        self.limit = power;
        Ok(())
    }
    fn get_power_management_limit(&mut self) -> Result<u32, ZeusdError> {
        Ok(self.limit)
    }
    fn get_power_management_limit_constraints(&mut self) -> Result<(u32, u32), ZeusdError> {
        Ok((100_000, 300_000))
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

async fn operation(request: RequestBuilder) -> OperationSnapshot {
    let response = request.send().await.unwrap();
    assert!(matches!(
        response.status(),
        StatusCode::OK | StatusCode::ACCEPTED
    ));
    assert_eq!(response.headers()["cache-control"], "no-store");
    response.json().await.unwrap()
}

async fn wait(
    client: &Client,
    base: &str,
    operation: &OperationSnapshot,
    ready: impl Fn(&OperationSnapshot) -> bool,
) -> OperationSnapshot {
    tokio::time::timeout(TIMEOUT, async {
        loop {
            let response = client
                .get(format!("{base}{}", operation.status_url))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let snapshot: OperationSnapshot = response.json().await.unwrap();
            if ready(&snapshot) {
                return snapshot;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap()
}

struct Harness {
    base: String,
    client: Client,
    gates: Vec<Arc<Gate>>,
    calls: Arc<Mutex<Vec<(usize, u32)>>>,
    server: ServerHandle,
}

impl Harness {
    async fn start(limits: OperationLimits) -> Self {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let gates: Vec<_> = (0..2).map(|_| Arc::new(Gate::default())).collect();
        let tasks = web::Data::new(
            GpuManagementTasks::start(
                gates
                    .iter()
                    .enumerate()
                    .map(|(id, gate)| FakeGpu {
                        id,
                        limit: 300_000,
                        gate: gate.clone(),
                        calls: calls.clone(),
                    })
                    .collect(),
            )
            .unwrap(),
        );
        let registry = web::Data::new(OperationRegistry::new(limits));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = HttpServer::new(move || {
            App::new()
                .app_data(tasks.clone())
                .app_data(registry.clone())
                .service(web::scope("/gpu").configure(gpu_control_routes))
        })
        .workers(2)
        .listen(listener)
        .unwrap()
        .run();
        let handle = server.handle();
        tokio::spawn(server);
        Self {
            base,
            client: Client::builder().timeout(TIMEOUT).build().unwrap(),
            gates,
            calls,
            server: handle,
        }
    }

    fn post(&self, suffix: &str) -> RequestBuilder {
        self.client
            .post(format!("{}{OPERATIONS}{suffix}", self.base))
    }

    fn activate(&self, op: &OperationSnapshot) -> RequestBuilder {
        self.client
            .post(format!("{}{}/activate", self.base, op.status_url))
    }

    fn cancel(&self, op: &OperationSnapshot) -> RequestBuilder {
        self.client
            .delete(format!("{}{}", self.base, op.status_url))
    }

    async fn wait(
        &self,
        op: &OperationSnapshot,
        ready: impl Fn(&OperationSnapshot) -> bool,
    ) -> OperationSnapshot {
        wait(&self.client, &self.base, op, ready).await
    }

    async fn discard_reply(&self, path: &str) {
        let mut stream = tokio::net::TcpStream::connect(self.base.strip_prefix("http://").unwrap())
            .await
            .unwrap();
        let request = format!("POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        stream.write_all(request.as_bytes()).await.unwrap();
        // Wait for server acceptance, then discard the reply before receiving its ID or result.
        let mut first = [0; 1];
        tokio::time::timeout(TIMEOUT, stream.read_exact(&mut first))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(&first, b"H");
    }

    fn assert_calls(&self, expected: [&[u32]; 2]) {
        let calls = self.calls.lock().unwrap();
        for (gpu, expected) in expected.into_iter().enumerate() {
            let actual: Vec<_> = calls
                .iter()
                .filter(|(id, _)| *id == gpu)
                .map(|(_, power)| *power)
                .collect();
            assert_eq!(actual, expected, "GPU {gpu}");
        }
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
async fn lost_activation_reply_and_concurrent_retries_execute_each_device_once() {
    let h = Harness::start(OperationLimits::default()).await;
    let prepared =
        operation(h.post("/prepare?gpu_ids=0,1&power_limit_mw=200000&start_within_ms=60000")).await;
    assert!(prepared
        .devices
        .values()
        .all(|d| d.state == DeviceState::Prepared));
    h.assert_calls([&[], &[]]);
    h.discard_reply(&format!("{}/activate", prepared.status_url))
        .await;
    h.wait(&prepared, |s| {
        s.devices.values().all(|d| d.state == DeviceState::Running)
    })
    .await;
    let retries = futures::future::join_all((0..4).map(|_| operation(h.activate(&prepared)))).await;
    for retry in retries {
        assert_eq!(retry.id, prepared.id);
        assert!(retry
            .devices
            .values()
            .all(|d| d.state == DeviceState::Running));
    }
    let cancelled = operation(h.cancel(&prepared)).await;
    assert!(!cancelled.complete);
    assert!(cancelled
        .devices
        .values()
        .all(|d| d.state == DeviceState::Running));
    h.gates[0].release();
    let partial = h
        .wait(&prepared, |s| s.devices[&0].state == DeviceState::Succeeded)
        .await;
    assert_eq!(partial.devices[&1].state, DeviceState::Running);
    h.gates[1].release();
    h.wait(&prepared, |s| s.complete).await;
    assert!(operation(h.activate(&prepared)).await.complete);
    let following =
        operation(h.post("/submit?gpu_ids=0,1&power_limit_mw=260000&start_within_ms=60000")).await;
    h.wait(&following, |s| s.complete).await;
    h.assert_calls([&[200_000, 260_000], &[200_000, 260_000]]);
}

#[tokio::test]
async fn cancelled_and_expired_commands_never_execute_after_the_queue_unblocks() {
    let h = Harness::start(OperationLimits {
        completed: 1,
        ..OperationLimits::default()
    })
    .await;
    let legacy = h
        .client
        .post(format!(
            "{}/gpu/set_power_limit?gpu_ids=0&power_limit_mw=200000&block=false",
            h.base
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(legacy.status(), StatusCode::OK);
    let prepared =
        operation(h.post("/prepare?gpu_ids=0,1&power_limit_mw=230000&start_within_ms=60000")).await;
    assert!(operation(h.cancel(&prepared)).await.complete);
    let replay = operation(h.activate(&prepared)).await;
    assert!(replay
        .devices
        .values()
        .all(|d| d.state == DeviceState::Cancelled && d.effect == SettingEffect::Unchanged));

    let partial =
        operation(h.post("/prepare?gpu_ids=0,1&power_limit_mw=250000&start_within_ms=60000")).await;
    futures::future::join_all((0..4).map(|_| operation(h.activate(&partial)))).await;
    h.wait(&partial, |s| s.devices[&1].state == DeviceState::Succeeded)
        .await;
    let cancelled = operation(h.cancel(&partial)).await;
    assert_eq!(cancelled.devices[&0].state, DeviceState::Cancelled);
    assert_eq!(cancelled.devices[&0].effect, SettingEffect::Unchanged);
    assert_eq!(cancelled.devices[&1].effect, SettingEffect::Applied);
    assert_eq!(
        h.activate(&prepared).send().await.unwrap().status(),
        StatusCode::GONE
    );

    for suffix in ["/submit", "/prepare"] {
        let expired = operation(h.post(&format!(
            "{suffix}?gpu_ids=0&power_limit_mw=240000&start_within_ms=20"
        )))
        .await;
        let done = h.wait(&expired, |s| s.complete).await;
        assert_eq!(
            done.devices[&0].reason,
            Some(OutcomeReason::DeadlineExceeded)
        );
        assert_eq!(done.devices[&0].effect, SettingEffect::Unchanged);
        assert!(operation(h.activate(&expired)).await.complete);
    }
    h.gates[0].release();
    let following =
        operation(h.post("/submit?gpu_ids=0,1&power_limit_mw=260000&start_within_ms=60000")).await;
    h.wait(&following, |s| s.complete).await;
    h.assert_calls([&[200_000, 260_000], &[250_000, 260_000]]);
}

#[tokio::test]
async fn abandoned_preparation_expires_and_old_ids_cannot_execute_after_restart() {
    let h = Harness::start(OperationLimits {
        live: 1,
        device_results: 2,
        ..OperationLimits::default()
    })
    .await;
    let invalid = h
        .post("/prepare?gpu_ids=0,99&power_limit_mw=250000&start_within_ms=60000")
        .send()
        .await
        .unwrap();
    assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
    h.discard_reply(&format!(
        "{OPERATIONS}/prepare?gpu_ids=0,1&power_limit_mw=230000&start_within_ms=1000"
    ))
    .await;
    let rejected = h
        .post("/submit?gpu_ids=0,1&power_limit_mw=250000&start_within_ms=60000")
        .send()
        .await
        .unwrap();
    assert_eq!(rejected.status(), StatusCode::SERVICE_UNAVAILABLE);
    h.assert_calls([&[], &[]]);
    let replacement: OperationSnapshot = tokio::time::timeout(TIMEOUT, async {
        loop {
            let response = h
                .post("/prepare?gpu_ids=0,1&power_limit_mw=250000&start_within_ms=60000")
                .send()
                .await
                .unwrap();
            if response.status() == StatusCode::ACCEPTED {
                return response.json().await.unwrap();
            }
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    h.assert_calls([&[], &[]]);
    operation(h.activate(&replacement)).await;
    h.wait(&replacement, |s| s.complete).await;
    h.assert_calls([&[250_000], &[250_000]]);
    h.server.stop(false).await;
    let restarted = Harness::start(OperationLimits::default()).await;
    let unavailable = restarted.activate(&replacement).send().await.unwrap();
    assert_eq!(unavailable.status(), StatusCode::GONE);
    let error: serde_json::Value = unavailable.json().await.unwrap();
    assert_eq!(error["error"]["effect"], "uncertain");
    restarted.assert_calls([&[], &[]]);
}

#[tokio::test]
async fn native_failure_preserves_uncertainty_and_executor_death_cancels_queued_work() {
    let h = Harness::start(OperationLimits::default()).await;
    let failure =
        operation(h.post("/prepare?gpu_ids=0&power_limit_mw=999000&start_within_ms=60000")).await;
    operation(h.activate(&failure)).await;
    let failed = h.wait(&failure, |s| s.complete).await;
    assert_eq!(failed.devices[&0].effect, SettingEffect::Uncertain);
    assert_eq!(failed.devices[&0].reason, Some(OutcomeReason::BackendError));
    assert!(operation(h.activate(&failure)).await.complete);

    let running =
        operation(h.post("/submit?gpu_ids=1&power_limit_mw=200000&start_within_ms=60000")).await;
    h.wait(&running, |s| s.devices[&1].state == DeviceState::Running)
        .await;
    let panic =
        operation(h.post("/submit?gpu_ids=1&power_limit_mw=666000&start_within_ms=60000")).await;
    let queued =
        operation(h.post("/submit?gpu_ids=1&power_limit_mw=250000&start_within_ms=60000")).await;
    h.gates[1].release();
    let failed = h.wait(&panic, |s| s.complete).await;
    assert_eq!(
        failed.devices[&1].reason,
        Some(OutcomeReason::ExecutorPanicked)
    );
    assert_eq!(failed.devices[&1].effect, SettingEffect::Uncertain);
    let cancelled = h.wait(&queued, |s| s.complete).await;
    assert_eq!(
        cancelled.devices[&1].reason,
        Some(OutcomeReason::ExecutorUnavailable)
    );
    assert_eq!(cancelled.devices[&1].effect, SettingEffect::Unchanged);
    let following =
        operation(h.post("/submit?gpu_ids=0,1&power_limit_mw=260000&start_within_ms=60000")).await;
    let done = h.wait(&following, |s| s.complete).await;
    assert_eq!(done.devices[&0].effect, SettingEffect::Applied);
    assert_eq!(done.devices[&1].effect, SettingEffect::Unchanged);
    h.assert_calls([&[999_000, 260_000], &[200_000, 666_000]]);
}

#[tokio::test]
async fn cpu_activation_uses_the_saved_constraint_and_reports_each_outcome() {
    let app = helpers::TestApp::start_with_cpu_write_errnos(&[Some(5), None]).await;
    let client = Client::builder().timeout(TIMEOUT).build().unwrap();
    let base = format!("http://127.0.0.1:{}", app.port);
    let read = format!("{base}/cpu/get_power_limit");
    let original: serde_json::Value = client
        .get(&read)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let prepared = operation(client.post(format!("{base}/cpu/set_power_limit/operations/prepare?cpu_ids=0,1&constraint=long_term&power_limit_mw=150000&start_within_ms=60000"))).await;
    let unchanged: serde_json::Value = client
        .get(&read)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(unchanged, original);
    let activate = format!("{base}{}/activate", prepared.status_url);
    operation(client.post(&activate)).await;
    let done = wait(&client, &base, &prepared, |s| s.complete).await;
    assert_eq!(done.devices[&0].effect, SettingEffect::Uncertain);
    assert_eq!(done.devices[&1].effect, SettingEffect::Applied);
    assert!(operation(client.post(&activate)).await.complete);
    let mut expected = original;
    let constraint = expected["1"]["cpu"]["constraints"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|c| c["name"] == "long_term")
        .unwrap();
    constraint["power_limit_mw"] = serde_json::json!(150_000);
    let actual: serde_json::Value = client
        .get(&read)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(actual, expected);
}

#[tokio::test]
async fn tracked_operations_require_control_permission_throughout_the_lifecycle() {
    const KEY: &[u8] = b"operations-test-signing-key";
    let app = helpers::TestApp::start_with_auth(KEY).await;
    let client = Client::builder().timeout(TIMEOUT).build().unwrap();
    let base = format!("http://127.0.0.1:{}", app.port);
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
        let root = format!("{base}/{family}/set_power_limit/operations");
        let prepared = operation(
            client
                .post(format!("{root}/prepare?{query}"))
                .bearer_auth(&writer),
        )
        .await;
        let status = format!("{base}{}", prepared.status_url);
        let activate = format!("{status}/activate");
        for (method, url) in [
            (Method::POST, format!("{root}/submit?{query}")),
            (Method::POST, format!("{root}/prepare?{query}")),
            (Method::POST, activate.clone()),
            (Method::GET, status.clone()),
            (Method::DELETE, status.clone()),
        ] {
            let rejected = client
                .request(method.clone(), &url)
                .bearer_auth(&reader)
                .send()
                .await
                .unwrap();
            assert_eq!(rejected.status(), StatusCode::FORBIDDEN);
            assert_eq!(
                client.request(method, &url).send().await.unwrap().status(),
                StatusCode::UNAUTHORIZED
            );
        }
        let unchanged = operation(client.get(&status).bearer_auth(&writer)).await;
        assert!(unchanged
            .devices
            .values()
            .all(|d| d.state == DeviceState::Prepared));
        operation(client.post(&activate).bearer_auth(&writer)).await;
        let cancelled = operation(client.delete(&status).bearer_auth(&writer)).await;
        assert_eq!(cancelled.id, prepared.id);
    }
}
