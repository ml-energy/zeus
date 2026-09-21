mod helpers;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio_stream::StreamExt;
use zeusd::devices::cpu::power::start_cpu_poller;
use zeusd::devices::cpu::RaplResponse;
use zeusd::devices::cpu::{CpuManager, PackageInfo};
use zeusd::error::ZeusdError;
use zeusd::routes::cpu::GetCumulativeEnergy;

use crate::helpers::TestApp;

#[tokio::test]
async fn test_only_cpu_measuremnt() {
    let mut app = TestApp::start().await;
    let measurements: Vec<u64> = vec![10000, 10001, 12313, 8213, 0];
    app.set_cpu_energy_measurements(0, &measurements);

    for expected in measurements {
        let resp = app
            .send(GetCumulativeEnergy {
                cpu_ids: Some("0".to_string()),
                cpu: true,
                dram: false,
            })
            .await
            .expect("Failed to send request");
        assert_eq!(resp.status(), 200);
        let response_map: HashMap<String, RaplResponse> =
            serde_json::from_str(&resp.text().await.unwrap())
                .expect("Failed to deserialize response body");
        let rapl_response = response_map.get("0").expect("Missing CPU 0 in response");
        assert_eq!(rapl_response.cpu_energy_uj.unwrap(), expected);
        assert_eq!(rapl_response.dram_energy_uj, None);
    }
}

#[tokio::test]
async fn test_only_dram_measuremnt() {
    let mut app = TestApp::start().await;
    let measurements: Vec<u64> = vec![10000, 10001, 12313, 8213, 0];
    app.set_dram_energy_measurements(0, &measurements);

    for expected in measurements {
        let resp = app
            .send(GetCumulativeEnergy {
                cpu_ids: Some("0".to_string()),
                cpu: false,
                dram: true,
            })
            .await
            .expect("Failed to send request");
        assert_eq!(resp.status(), 200);
        let response_map: HashMap<String, RaplResponse> =
            serde_json::from_str(&resp.text().await.unwrap())
                .expect("Failed to deserialize response body");
        let rapl_response = response_map.get("0").expect("Missing CPU 0 in response");
        assert_eq!(rapl_response.cpu_energy_uj, None);
        assert_eq!(rapl_response.dram_energy_uj.unwrap(), expected);
    }
}

#[tokio::test]
async fn test_both_measuremnt() {
    let mut app = TestApp::start().await;
    let measurements: Vec<u64> = vec![10000, 10001, 12313, 8213, 0];
    app.set_cpu_energy_measurements(0, &measurements);
    app.set_dram_energy_measurements(0, &measurements);

    for expected in measurements {
        let resp = app
            .send(GetCumulativeEnergy {
                cpu_ids: Some("0".to_string()),
                cpu: true,
                dram: true,
            })
            .await
            .expect("Failed to send request");
        assert_eq!(resp.status(), 200);
        let response_map: HashMap<String, RaplResponse> =
            serde_json::from_str(&resp.text().await.unwrap())
                .expect("Failed to deserialize response body");
        let rapl_response = response_map.get("0").expect("Missing CPU 0 in response");
        assert_eq!(rapl_response.cpu_energy_uj.unwrap(), expected);
        assert_eq!(rapl_response.dram_energy_uj.unwrap(), expected);
    }
}

#[tokio::test]
async fn test_invalid_requests() {
    let app = TestApp::start().await;

    let client = reqwest::Client::new();

    // Missing dram field
    let url = format!(
        "http://127.0.0.1:{}/cpu/get_cumulative_energy?cpu_ids=0&cpu=true",
        app.port
    );
    let resp = client
        .get(url)
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);

    // Missing cpu field
    let url = format!(
        "http://127.0.0.1:{}/cpu/get_cumulative_energy?cpu_ids=0&dram=true",
        app.port
    );
    let resp = client
        .get(url)
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);

    // Invalid type
    let url = format!(
        "http://127.0.0.1:{}/cpu/get_cumulative_energy?cpu_ids=0&cpu=notabool&dram=true",
        app.port
    );
    let resp = client
        .get(url)
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);

    // Invalid field name + out of index CPU
    let url = format!(
        "http://127.0.0.1:{}/cpu/get_cumulative_energy?cpu_ids=2&cp=true&dram=true",
        app.port
    );
    let resp = client
        .get(url)
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);

    // Out of index CPU
    let url = format!(
        "http://127.0.0.1:{}/cpu/get_cumulative_energy?cpu_ids=2&cpu=true&dram=true",
        app.port
    );
    let resp = client
        .get(url)
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn test_cpu_power_oneshot() {
    use crate::helpers::{
        POWER_TEST_CPU_INCREMENT_UJ, POWER_TEST_DRAM_INCREMENT_UJ, POWER_TEST_POLL_HZ,
    };

    let app = TestApp::start().await;
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/cpu/get_power", app.port);
    let resp = client
        .get(&url)
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("Failed to parse JSON");

    assert!(body["timestamp_ms"].is_number());
    assert!(body["timestamp_ms"].as_u64().unwrap() > 0);

    let power_mw = &body["power_mw"];
    assert!(power_mw.is_object());

    let cpu0 = &power_mw["0"];
    assert!(cpu0.is_object(), "Expected CPU 0 power data, got: {body}");

    // The mock advances energy by a fixed increment per read, so the delta is
    // exact, but power is the delta over the measured elapsed time, which is
    // at least the configured sampling period.
    let period_us = 1_000_000u64 / POWER_TEST_POLL_HZ as u64;
    let max_cpu_mw = POWER_TEST_CPU_INCREMENT_UJ * 1000 / period_us;
    let max_dram_mw = POWER_TEST_DRAM_INCREMENT_UJ * 1000 / period_us;

    let cpu_mw = cpu0["cpu_mw"].as_u64().unwrap();
    let dram_mw = cpu0["dram_mw"].as_u64().unwrap();
    assert!(
        cpu_mw > 0 && cpu_mw <= max_cpu_mw,
        "Expected CPU power in (0, {max_cpu_mw}] mW, got {cpu_mw}"
    );
    assert!(
        dram_mw > 0 && dram_mw <= max_dram_mw,
        "Expected DRAM power in (0, {max_dram_mw}] mW, got {dram_mw}"
    );
}

#[tokio::test]
async fn test_cpu_power_stream_receives_events() {
    let _app = TestApp::start().await;
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/cpu/stream_power?cpu_ids=0", _app.port);
    let mut resp = client
        .get(&url)
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .expect("Missing content-type")
            .to_str()
            .unwrap(),
        "text/event-stream"
    );

    let chunk = tokio::time::timeout(tokio::time::Duration::from_secs(2), resp.chunk())
        .await
        .expect("Timed out waiting for CPU power event")
        .expect("Failed to read CPU power event")
        .expect("CPU power stream ended before first event");
    let event = std::str::from_utf8(&chunk).expect("CPU power event should be UTF-8");
    let json = event
        .strip_prefix("data: ")
        .and_then(|event| event.strip_suffix("\n\n"))
        .expect("CPU power event should be an SSE data event");
    let body: serde_json::Value =
        serde_json::from_str(json).expect("CPU power event should contain JSON");
    assert!(body["timestamp_ms"].is_number());
    assert_eq!(body["cpu_id"].as_u64().unwrap(), 0);
    assert!(body["cpu_mw"].is_number());
    assert!(body["dram_mw"].is_number());
}

struct PollCountingCpu {
    poll_count: Arc<AtomicUsize>,
    cpu_energy_uj: u64,
    dram_energy_uj: u64,
    fail_cpu_read_number: Option<usize>,
    fail_first_dram_read: bool,
}

impl CpuManager for PollCountingCpu {
    fn device_count() -> Result<usize, ZeusdError> {
        Ok(1)
    }

    fn get_available_fields(
        index: usize,
    ) -> Result<(Arc<PackageInfo>, Option<Arc<PackageInfo>>), ZeusdError> {
        Ok((
            Arc::new(PackageInfo {
                index,
                name: "package-0".to_string(),
                energy_uj_path: PathBuf::from(
                    "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
                ),
                max_energy_uj: 1_000_000,
            }),
            Some(Arc::new(PackageInfo {
                index,
                name: "dram".to_string(),
                energy_uj_path: PathBuf::from(
                    "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj",
                ),
                max_energy_uj: 1_000_000,
            })),
        ))
    }

    fn get_cpu_energy(&mut self) -> Result<u64, ZeusdError> {
        let read_number = self.poll_count.fetch_add(1, Ordering::Relaxed);
        if self.fail_cpu_read_number == Some(read_number) {
            // Model energy continuing to accumulate while this read is lost.
            self.cpu_energy_uj += 10_000;
            return Err(ZeusdError::CpuPowerMeasurementError(0));
        }

        let value = self.cpu_energy_uj;
        self.cpu_energy_uj += 10_000;
        Ok(value)
    }

    fn get_dram_energy(&mut self) -> Result<u64, ZeusdError> {
        if std::mem::take(&mut self.fail_first_dram_read) {
            return Err(ZeusdError::CpuPowerMeasurementError(0));
        }

        let value = self.dram_energy_uj;
        self.dram_energy_uj += 5_000;
        Ok(value)
    }

    fn is_dram_available(&self) -> bool {
        true
    }
}

#[tokio::test]
async fn test_cpu_power_polls_only_subscribed_cpu() {
    let poll_count_0 = Arc::new(AtomicUsize::new(0));
    let poll_count_1 = Arc::new(AtomicUsize::new(0));
    let cpu_0 = PollCountingCpu {
        poll_count: poll_count_0.clone(),
        cpu_energy_uj: 0,
        dram_energy_uj: 0,
        fail_cpu_read_number: None,
        fail_first_dram_read: false,
    };
    let cpu_1 = PollCountingCpu {
        poll_count: poll_count_1.clone(),
        cpu_energy_uj: 0,
        dram_energy_uj: 0,
        fail_cpu_read_number: None,
        fail_first_dram_read: false,
    };
    let broadcasts = start_cpu_poller(vec![(0, cpu_0), (1, cpu_1)], 100);
    let broadcast_1 = broadcasts.get(1).expect("Missing CPU 1 broadcast");

    let guard = broadcast_1.add_subscriber();
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    assert_eq!(poll_count_0.load(Ordering::Relaxed), 0);
    assert!(poll_count_1.load(Ordering::Relaxed) > 0);

    drop(guard);
}

#[tokio::test]
async fn test_cpu_power_stream_uses_full_elapsed_time_after_read_failure() {
    let cpu = PollCountingCpu {
        poll_count: Arc::new(AtomicUsize::new(0)),
        cpu_energy_uj: 0,
        dram_energy_uj: 0,
        // Read 0 establishes the baseline. Read 1 fails after another
        // interval of energy has accumulated; read 2 then spans two ticks.
        fail_cpu_read_number: Some(1),
        fail_first_dram_read: false,
    };
    // Use a deliberately slow rate so the test task has ample time to observe
    // each watch update rather than relying on coalescing behavior.
    let broadcasts = start_cpu_poller(vec![(0, cpu)], 10);
    let broadcast = broadcasts.get(0).expect("Missing CPU 0 broadcast");

    let mut stream = Box::pin(broadcast.stream());
    let guard = broadcast.add_subscriber();

    // The failed CPU read still produces a snapshot because DRAM advances.
    // Synchronizing on it makes the next stream item the recovery sample.
    let failed_read_sample =
        tokio::time::timeout(tokio::time::Duration::from_secs(2), stream.next())
            .await
            .expect("Timed out waiting for failed-read snapshot")
            .expect("CPU power stream ended before failed-read snapshot");
    assert_eq!(failed_read_sample.cpu_mw, 0);

    let recovered_sample = tokio::time::timeout(tokio::time::Duration::from_secs(2), stream.next())
        .await
        .expect("Timed out waiting for CPU power recovery")
        .expect("CPU power stream ended before recovery sample");

    // The mock accumulates 20,000 uJ across roughly two 100 ms intervals, so
    // the correct result is about 100 mW. The old nominal-period math reports
    // exactly 200 mW regardless of the elapsed time.
    assert!(
        recovered_sample.cpu_mw > 0 && recovered_sample.cpu_mw <= 150,
        "post-failure power should use the full elapsed interval; got {} mW",
        recovered_sample.cpu_mw,
    );
    drop(guard);
}

#[tokio::test]
async fn test_cpu_power_stream_recovers_dram_after_initial_read_failure() {
    let cpu = PollCountingCpu {
        poll_count: Arc::new(AtomicUsize::new(0)),
        cpu_energy_uj: 0,
        dram_energy_uj: 0,
        fail_cpu_read_number: None,
        fail_first_dram_read: true,
    };
    let broadcasts = start_cpu_poller(vec![(0, cpu)], 100);
    let broadcast = broadcasts.get(0).expect("Missing CPU 0 broadcast");

    // Create the stream before waking the poller so the first transition after
    // the failed DRAM baseline cannot be missed.
    let mut stream = Box::pin(broadcast.stream());
    let guard = broadcast.add_subscriber();

    let recovered = tokio::time::timeout(tokio::time::Duration::from_secs(2), async {
        while let Some(sample) = stream.next().await {
            if sample.dram_mw.is_some() {
                return true;
            }
        }
        false
    })
    .await
    .expect("Timed out waiting for DRAM power recovery");

    assert!(
        recovered,
        "DRAM power should recover after the initial baseline read fails"
    );
    drop(guard);
}

#[tokio::test]
async fn test_deny_unknown_query_fields() {
    let app = TestApp::start().await;
    let client = reqwest::Client::new();

    // cpu/get_power with gpu_ids (wrong query field) should be rejected.
    let url = format!("http://127.0.0.1:{}/cpu/get_power?gpu_ids=0", app.port);
    let resp = client
        .get(&url)
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);

    // gpu/get_power with cpu_ids (wrong query field) should be rejected.
    let url = format!("http://127.0.0.1:{}/gpu/get_power?cpu_ids=0", app.port);
    let resp = client
        .get(&url)
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);
}
