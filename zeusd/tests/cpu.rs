mod helpers;

use std::collections::HashMap;
use zeusd::devices::cpu::RaplResponse;
use zeusd::routes::cpu::GetCumulativeEnergy;

use crate::helpers::{TestApp, ZeusdRequest};

#[tokio::test]
async fn test_only_cpu_measuremnt() {
    let mut app = TestApp::start().await;
    let measurements: Vec<u64> = vec![10000, 10001, 12313, 8213, 0];
    app.set_cpu_energy_measurements(0, &measurements);

    for expected in measurements {
        let resp = app
            .send(GetCumulativeEnergy {
                cpu_ids: "0".to_string(),
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
                cpu_ids: "0".to_string(),
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
                cpu_ids: "0".to_string(),
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

    // CPU 0 should have deterministic power based on the test constants.
    let cpu0 = &power_mw["0"];
    assert!(cpu0.is_object(), "Expected CPU 0 power data, got: {body}");

    let period_us = 1_000_000u64 / POWER_TEST_POLL_HZ as u64;
    let expected_cpu_mw = POWER_TEST_CPU_INCREMENT_UJ * 1000 / period_us;
    let expected_dram_mw = POWER_TEST_DRAM_INCREMENT_UJ * 1000 / period_us;

    assert_eq!(cpu0["cpu_mw"].as_u64().unwrap(), expected_cpu_mw);
    assert_eq!(cpu0["dram_mw"].as_u64().unwrap(), expected_dram_mw);
}

#[tokio::test]
async fn test_cpu_power_stream_receives_events() {
    let _app = TestApp::start().await;
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/cpu/stream_power", _app.port);
    let resp = client
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
