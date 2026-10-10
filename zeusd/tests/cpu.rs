mod helpers;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio_stream::StreamExt;
use zeusd::devices::cpu::power::start_cpu_poller;
use zeusd::devices::cpu::RaplResponse;
use zeusd::devices::cpu::{CpuDramPowerLimits, CpuManager, CpuPowerLimitConstraints, PackageInfo};
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
async fn test_get_power_limit() {
    let app = TestApp::start().await;
    let client = reqwest::Client::new();
    let expected = serde_json::json!({
        "0": {
            "cpu": {
                "enabled": true,
                "constraints": [
                    {"name": "long_term", "power_limit_mw": 205000, "max_power_mw": 205000, "time_window_us": 999424},
                    {"name": "short_term", "power_limit_mw": 246000, "max_power_mw": 780000, "time_window_us": 999424},
                    {"name": "peak_power", "power_limit_mw": 300000, "max_power_mw": 1560000, "time_window_us": null},
                ],
            },
            "dram": {
                "enabled": false,
                "constraints": [
                    {"name": "long_term", "power_limit_mw": 0, "max_power_mw": 121000, "time_window_us": 976},
                ],
            },
        },
    });

    for query in ["", "?cpu_ids=0"] {
        let url = format!("http://127.0.0.1:{}/cpu/get_power_limit{query}", app.port);
        let resp = client
            .get(&url)
            .send()
            .await
            .expect("Failed to send request");
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.expect("Failed to parse JSON");
        assert_eq!(body, expected);
    }

    // Out of index CPU and unknown query field.
    for query in ["?cpu_ids=1", "?gpu_ids=0"] {
        let url = format!("http://127.0.0.1:{}/cpu/get_power_limit{query}", app.port);
        let resp = client
            .get(&url)
            .send()
            .await
            .expect("Failed to send request");
        assert_eq!(resp.status(), 400);
    }
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

/// A stream of a CPU whose energy counter Zeusd cannot read fails with the
/// cause instead of starting a stream that never sends a sample.
#[tokio::test]
async fn test_cpu_power_stream_rejects_unreadable_energy_counter() {
    // CPU 1 cannot read its CPU energy counter, CPU 2 only its DRAM counter.
    let app = TestApp::start_with_test_cpus(3, |index, cpu| {
        cpu.cpu_energy_denied = index == 1;
        cpu.dram_energy_denied = index == 2;
    })
    .await;
    let client = reqwest::Client::new();
    let stream_url = |query: &str| format!("http://127.0.0.1:{}/cpu/stream_power{query}", app.port);

    for query in ["?cpu_ids=1", "?cpu_ids=0,1", ""] {
        let resp = client
            .get(stream_url(query))
            .send()
            .await
            .expect("Failed to send request");
        assert_eq!(resp.status(), 403, "{query}");
        let body: serde_json::Value = resp.json().await.expect("Failed to parse JSON");
        let errors = body["errors"].as_object().unwrap();
        assert_eq!(errors.keys().collect::<Vec<_>>(), vec!["1"], "{query}");
    }

    for query in ["?cpu_ids=0", "?cpu_ids=2"] {
        let mut resp = client
            .get(stream_url(query))
            .send()
            .await
            .expect("Failed to send request");
        assert_eq!(resp.status(), 200, "{query}");
        tokio::time::timeout(tokio::time::Duration::from_secs(2), resp.chunk())
            .await
            .expect("Timed out waiting for CPU power event")
            .expect("Failed to read CPU power event")
            .expect("CPU power stream ended before first event");
    }

    // CPU-only energy requests work when only the DRAM counter is unreadable.
    let resp = client
        .get(format!(
            "http://127.0.0.1:{}/cpu/get_cumulative_energy?cpu_ids=2&cpu=true&dram=false",
            app.port
        ))
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 200);
    let resp = client
        .get(format!(
            "http://127.0.0.1:{}/cpu/get_cumulative_energy?cpu_ids=2&cpu=false&dram=true",
            app.port
        ))
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 403);
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
                zone_dir: PathBuf::from("/sys/class/powercap/intel-rapl/intel-rapl:0"),
                energy_uj_path: PathBuf::from(
                    "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
                ),
                max_energy_uj: 1_000_000,
            }),
            Some(Arc::new(PackageInfo {
                index,
                name: "dram".to_string(),
                zone_dir: PathBuf::from(
                    "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0",
                ),
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

    fn get_power_limits(&self) -> Result<CpuDramPowerLimits, ZeusdError> {
        unimplemented!("The power poller does not read power limits")
    }

    fn get_power_limit_constraints(&self) -> Result<CpuPowerLimitConstraints, ZeusdError> {
        unimplemented!("The power poller does not read power limits")
    }

    fn set_power_limit(
        &mut self,
        _constraint: &str,
        _power_limit_mw: u64,
    ) -> Result<(), ZeusdError> {
        unimplemented!("The power poller does not control power limits")
    }

    fn set_power_limit_time_window(
        &mut self,
        _constraint: &str,
        _time_window_us: u64,
    ) -> Result<(), ZeusdError> {
        unimplemented!("The power poller does not control power limits")
    }

    fn reset_power_limits(&mut self) -> Result<(), ZeusdError> {
        unimplemented!("The power poller does not control power limits")
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

/// POST a CPU control endpoint and return the status code.
async fn post_cpu_control(app: &TestApp, endpoint_and_query: &str) -> u16 {
    reqwest::Client::new()
        .post(format!(
            "http://127.0.0.1:{}/cpu/{endpoint_and_query}",
            app.port
        ))
        .send()
        .await
        .expect("Failed to send request")
        .status()
        .as_u16()
}

/// GET the package zone constraints of CPU 0 as `(name, power_limit_mw, time_window_us)`.
async fn package_constraints(app: &TestApp) -> Vec<(String, u64, Option<u64>)> {
    let body: serde_json::Value = reqwest::Client::new()
        .get(format!("http://127.0.0.1:{}/cpu/get_power_limit", app.port))
        .send()
        .await
        .expect("Failed to send request")
        .json()
        .await
        .expect("Failed to parse JSON");
    body["0"]["cpu"]["constraints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            (
                c["name"].as_str().unwrap().to_string(),
                c["power_limit_mw"].as_u64().unwrap(),
                c["time_window_us"].as_u64(),
            )
        })
        .collect()
}

fn initial_package_constraints() -> Vec<(String, u64, Option<u64>)> {
    helpers::test_power_limits()
        .cpu
        .constraints
        .into_iter()
        .map(|c| (c.name, c.power_limit_mw, c.time_window_us))
        .collect()
}

#[tokio::test]
async fn test_cpu_set_power_limit() {
    let app = TestApp::start().await;

    let status = post_cpu_control(
        &app,
        "set_power_limit?cpu_ids=0&constraint=long_term&power_limit_mw=150000",
    )
    .await;
    assert_eq!(status, 200);

    let mut expected = initial_package_constraints();
    expected[0].1 = 150_000;
    assert_eq!(package_constraints(&app).await, expected);

    // RAPL does not bound limits by `max_power_mw`, which is TDP for `long_term`.
    let status = post_cpu_control(
        &app,
        "set_power_limit?cpu_ids=0&constraint=long_term&power_limit_mw=250000",
    )
    .await;
    assert_eq!(status, 200);
}

#[tokio::test]
async fn test_cpu_set_power_limit_invalid() {
    let app = TestApp::start().await;

    for query in [
        // Unknown constraint.
        "cpu_ids=0&constraint=socket&power_limit_mw=150000",
        // Zero.
        "cpu_ids=0&constraint=long_term&power_limit_mw=0",
        // CPU that does not exist.
        "cpu_ids=1&constraint=long_term&power_limit_mw=150000",
        // Empty CPU list.
        "cpu_ids=&constraint=long_term&power_limit_mw=150000",
        // Missing and unknown fields.
        "cpu_ids=0&constraint=long_term",
        "cpu_ids=0&constraint=long_term&power_limit_mw=150000&block=true",
        // Negative value.
        "cpu_ids=0&constraint=long_term&power_limit_mw=-1",
    ] {
        let status = post_cpu_control(&app, &format!("set_power_limit?{query}")).await;
        assert_eq!(status, 400, "{query} should be rejected");
    }

    assert_eq!(
        package_constraints(&app).await,
        initial_package_constraints()
    );
}

#[tokio::test]
async fn test_cpu_set_power_limit_time_window() {
    let app = TestApp::start().await;

    let status = post_cpu_control(
        &app,
        "set_power_limit_time_window?cpu_ids=0&constraint=short_term&time_window_us=2440",
    )
    .await;
    assert_eq!(status, 200);

    let mut expected = initial_package_constraints();
    expected[1].2 = Some(2_440);
    assert_eq!(package_constraints(&app).await, expected);

    for query in [
        // `peak_power` has no time window.
        "cpu_ids=0&constraint=peak_power&time_window_us=1000",
        "cpu_ids=0&constraint=long_term&time_window_us=0",
        "cpu_ids=0&constraint=socket&time_window_us=1000",
    ] {
        let status = post_cpu_control(&app, &format!("set_power_limit_time_window?{query}")).await;
        assert_eq!(status, 400, "{query} should be rejected");
    }
    assert_eq!(package_constraints(&app).await, expected);
}

#[tokio::test]
async fn test_cpu_reset_power_limit() {
    let app = TestApp::start().await;

    for endpoint_and_query in [
        "set_power_limit?cpu_ids=0&constraint=long_term&power_limit_mw=100000",
        "set_power_limit?cpu_ids=0&constraint=short_term&power_limit_mw=120000",
        "set_power_limit_time_window?cpu_ids=0&constraint=long_term&time_window_us=27983872",
    ] {
        assert_eq!(post_cpu_control(&app, endpoint_and_query).await, 200);
    }
    assert_ne!(
        package_constraints(&app).await,
        initial_package_constraints()
    );

    assert_eq!(
        post_cpu_control(&app, "reset_power_limit?cpu_ids=0").await,
        200
    );
    assert_eq!(
        package_constraints(&app).await,
        initial_package_constraints()
    );

    assert_eq!(
        post_cpu_control(&app, "reset_power_limit?cpu_ids=1").await,
        400
    );
}

#[tokio::test]
async fn test_cpu_control_only_mode() {
    let app = TestApp::start_with_groups(&[zeusd::config::ApiGroup::CpuControl]).await;
    let client = reqwest::Client::new();

    let status = post_cpu_control(
        &app,
        "set_power_limit?cpu_ids=0&constraint=long_term&power_limit_mw=150000",
    )
    .await;
    assert_eq!(status, 200);

    let resp = client
        .get(format!("http://127.0.0.1:{}/cpu/get_power_limit", app.port))
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_cpu_read_only_mode_rejects_control() {
    let app = TestApp::start_with_groups(&[zeusd::config::ApiGroup::CpuRead]).await;

    for endpoint_and_query in [
        "set_power_limit?cpu_ids=0&constraint=long_term&power_limit_mw=150000",
        "set_power_limit_time_window?cpu_ids=0&constraint=long_term&time_window_us=999424",
        "reset_power_limit?cpu_ids=0",
    ] {
        assert_eq!(post_cpu_control(&app, endpoint_and_query).await, 404);
    }
    assert_eq!(
        package_constraints(&app).await,
        initial_package_constraints()
    );
}

#[cfg(unix)]
#[tokio::test]
async fn test_cpu_control_partial_failure() {
    // CPU 1 fails with EACCES (a BIOS-locked limit), CPU 2 with EIO.
    let app = TestApp::start_with_cpu_write_errnos(&[None, Some(13), Some(5)]).await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!(
            "http://127.0.0.1:{}/cpu/set_power_limit?cpu_ids=0,1&constraint=long_term&power_limit_mw=150000",
            app.port
        ))
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 403);
    let body: serde_json::Value = resp.json().await.expect("Failed to parse JSON");
    let errors = body["errors"].as_object().unwrap();
    assert_eq!(errors.len(), 1);
    assert!(errors.contains_key("1"));

    // CPU 0 applied the limit despite CPU 1 failing.
    let body: serde_json::Value = client
        .get(format!(
            "http://127.0.0.1:{}/cpu/get_power_limit?cpu_ids=0",
            app.port
        ))
        .send()
        .await
        .expect("Failed to send request")
        .json()
        .await
        .expect("Failed to parse JSON");
    assert_eq!(
        body["0"]["cpu"]["constraints"][0]["power_limit_mw"],
        150_000
    );

    // The worst status across CPUs wins.
    let status = post_cpu_control(&app, "reset_power_limit?cpu_ids=0,1,2").await;
    assert_eq!(status, 500);
}

#[tokio::test]
async fn test_get_power_limit_constraints() {
    let app = TestApp::start().await;
    let client = reqwest::Client::new();
    let expected = serde_json::json!({
        "0": {
            "rapl": {
                "thermal_spec_power_mw": 205000,
                "min_power_mw": 113000,
                "max_power_mw": 780000,
                "max_time_window_us": 31981568,
                "power_limit_register_max_mw": 4095875,
            },
            "hsmp": null,
        },
    });

    for query in ["", "?cpu_ids=0"] {
        let resp = client
            .get(format!(
                "http://127.0.0.1:{}/cpu/get_power_limit_constraints{query}",
                app.port
            ))
            .send()
            .await
            .expect("Failed to send request");
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.expect("Failed to parse JSON");
        assert_eq!(body, expected);
    }

    let resp = client
        .get(format!(
            "http://127.0.0.1:{}/cpu/get_power_limit_constraints?cpu_ids=1",
            app.port
        ))
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);
}
