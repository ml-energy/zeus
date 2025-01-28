mod helpers;

use zeusd::devices::cpu::DramAvailabilityResponse;
use zeusd::devices::cpu::RaplResponse;
use zeusd::routes::cpu::GetIndexEnergy;

use crate::helpers::{TestApp, ZeusdRequest};

#[tokio::test]
async fn test_only_cpu_measuremnt() {
    let mut app = TestApp::start().await;
    let measurements: Vec<u64> = vec![10000, 10001, 12313, 8213, 0];
    app.set_cpu_energy_measurements(0, &measurements);

    for expected in measurements {
        let resp = app
            .send(
                0,
                GetIndexEnergy {
                    cpu: true,
                    dram: false,
                },
            )
            .await
            .expect("Failed to send request");
        assert_eq!(resp.status(), 200);
        let rapl_response: RaplResponse = serde_json::from_str(&resp.text().await.unwrap())
            .expect("Failed to deserialize response body");
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
            .send(
                0,
                GetIndexEnergy {
                    cpu: false,
                    dram: true,
                },
            )
            .await
            .expect("Failed to send request");
        assert_eq!(resp.status(), 200);
        let rapl_response: RaplResponse = serde_json::from_str(&resp.text().await.unwrap())
            .expect("Failed to deserialiez response body");
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
            .send(
                0,
                GetIndexEnergy {
                    cpu: true,
                    dram: true,
                },
            )
            .await
            .expect("Failed to send request");
        assert_eq!(resp.status(), 200);
        let rapl_response: RaplResponse = serde_json::from_str(&resp.text().await.unwrap())
            .expect("Failed to deserialiez response body");
        assert_eq!(rapl_response.cpu_energy_uj.unwrap(), expected);
        assert_eq!(rapl_response.dram_energy_uj.unwrap(), expected);
    }
}

#[tokio::test]
async fn test_invalid_requests() {
    let app = TestApp::start().await;

    let client = reqwest::Client::new();
    let url = GetIndexEnergy::build_url(&app, 0);
    let resp = client
        .post(url)
        .json(&serde_json::json!(
            {
                "cpu": true, // Missing dram field
            }
        ))
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);

    let url = GetIndexEnergy::build_url(&app, 0);
    let resp = client
        .post(url)
        .json(&serde_json::json!(
            {
                "dram": true, // Missing cpu field
            }
        ))
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);

    let url = GetIndexEnergy::build_url(&app, 0);
    let resp = client
        .post(url)
        .json(&serde_json::json!(
            {
                "cpu": "true", //Invalid type
                "dram": true,
            }
        ))
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);

    let url = GetIndexEnergy::build_url(&app, 2); // Out of index CPU
    let resp = client
        .post(url)
        .json(&serde_json::json!(
            {
                "cp": true, // Invalid field name
                "dram": true,
            }
        ))
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);

    let url = GetIndexEnergy::build_url(&app, 2); // Out of index CPU
    let resp = client
        .post(url)
        .json(&serde_json::json!(
            {
                "cpu": true,
                "dram": true,
            }
        ))
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn test_supports_dram_energy() {
    let app = TestApp::start().await;
    let url = format!("http://127.0.0.1:{}/cpu/0/supports_dram_energy", app.port);
    let client = reqwest::Client::new();

    let resp = client
        .get(url)
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 200);

    let dram_response: DramAvailabilityResponse = serde_json::from_str(&resp.text().await.unwrap())
        .expect("Failed to deserialize response body");
    assert_eq!(dram_response.dram_available, true);
}
