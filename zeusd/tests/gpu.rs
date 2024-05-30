mod helpers;

use std::collections::HashSet;
use tokio::task::JoinSet;
use zeusd::routes::gpu::{
    ResetGpuLockedClocks, ResetMemLockedClocks, SetGpuLockedClocks, SetMemLockedClocks,
    SetPersistenceMode, SetPowerLimit,
};

use crate::helpers::{TestApp, ZeusdRequest};

#[tokio::test]
async fn test_set_persistence_mode_single() {
    let mut app = TestApp::start().await;

    let resp = app
        .send(
            0,
            SetPersistenceMode {
                enabled: true,
                block: true,
            },
        )
        .await
        .expect("Failed to send request");

    assert_eq!(resp.status(), 200);
    let history = app.persistence_mode_history_for_gpu(0);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0], true);
}

#[tokio::test]
async fn test_set_persistence_mode_multiple() {
    let mut app = TestApp::start().await;

    let num_requests = 10;
    for i in 0..num_requests {
        let resp = app
            .send(
                i % 4,
                SetPersistenceMode {
                    enabled: (i / 4) % 2 == 0,
                    block: true,
                },
            )
            .await
            .expect("Failed to send request");

        assert_eq!(resp.status(), 200);
    }

    assert_eq!(
        app.persistence_mode_history_for_gpu(0),
        vec![true, false, true]
    );
    assert_eq!(
        app.persistence_mode_history_for_gpu(1),
        vec![true, false, true]
    );
    assert_eq!(app.persistence_mode_history_for_gpu(2), vec![true, false]);
    assert_eq!(app.persistence_mode_history_for_gpu(3), vec![true, false]);
}

#[tokio::test]
async fn test_set_persistence_mode_invalid() {
    let app = TestApp::start().await;

    let client = reqwest::Client::new();
    let url = SetPersistenceMode::build_url(&app, 0);
    let resp = client
        .post(url)
        .json(&serde_json::json!(
            {
                "disabled": false,  // Wrong field name
                "block": true
            }
        ))
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);
    assert!(resp
        .text()
        .await
        .expect("Failed to read response")
        .contains("missing field"));

    let url = SetPersistenceMode::build_url(&app, 0);
    let resp = client
        .post(url)
        .json(&serde_json::json!(
            {
                "enabled": 1,  // Invalid type
                "block": true
            }
        ))
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);
    assert!(resp
        .text()
        .await
        .expect("Failed to read response")
        .contains("invalid type"));

    for block in [true, false] {
        let url = SetPersistenceMode::build_url(&app, 5); // Invalid GPU ID
        let resp = client
            .post(url)
            .json(&serde_json::json!(
                {
                    "enabled": true,
                    "block": block
                }
            ))
            .send()
            .await
            .expect("Failed to send request");
        assert_eq!(resp.status(), 400);
    }
}

#[tokio::test]
async fn test_set_persistence_mode_bulk() {
    let mut app = TestApp::start().await;

    let mut set = JoinSet::new();
    for i in 0..10 {
        set.spawn(app.send(
            0,
            SetPersistenceMode {
                enabled: i % 3 == 0,
                block: false,
            },
        ));
    }
    let mut responses = Vec::with_capacity(10);
    for _ in 0..10 {
        responses.push(set.join_next().await);
    }
    for resp in responses.into_iter() {
        assert_eq!(
            resp.expect("Leaked future")
                .expect("Failed to join future")
                .expect("Failed to send request")
                .status(),
            200
        );
    }

    // After this blocking request finishes, all non-blocking ones should have completed.
    assert_eq!(
        app.send(
            0,
            SetPersistenceMode {
                enabled: false,
                block: true,
            },
        )
        .await
        .expect("Failed to send request")
        .status(),
        200
    );

    let history = app.persistence_mode_history_for_gpu(0);
    assert_eq!(history.len(), 11);
    assert_eq!(history.iter().filter(|enabled| **enabled).count(), 4);
    assert_eq!(history.iter().filter(|enabled| !**enabled).count(), 6 + 1);
}

#[tokio::test]
async fn test_set_power_limit_single() {
    let mut app = TestApp::start().await;

    let resp = app
        .send(
            0,
            SetPowerLimit {
                power_limit_mw: 100_000,
                block: true,
            },
        )
        .await
        .expect("Failed to send request");

    assert_eq!(resp.status(), 200);
    let history = app.power_limit_history_for_gpu(0);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0], 100_000);
}

#[tokio::test]
async fn test_set_power_limit_multiple() {
    let mut app = TestApp::start().await;

    let num_requests = 10;
    for i in 0..num_requests {
        let resp = app
            .send(
                i % 4,
                SetPowerLimit {
                    power_limit_mw: 100_000 + i * 10_000,
                    block: true,
                },
            )
            .await
            .expect("Failed to send request");

        assert_eq!(resp.status(), 200);
    }

    assert_eq!(
        app.power_limit_history_for_gpu(0),
        vec![100_000, 140_000, 180_000]
    );
    assert_eq!(
        app.power_limit_history_for_gpu(1),
        vec![110_000, 150_000, 190_000]
    );
    assert_eq!(app.power_limit_history_for_gpu(2), vec![120_000, 160_000]);
    assert_eq!(app.power_limit_history_for_gpu(3), vec![130_000, 170_000]);
}

#[tokio::test]
async fn test_set_power_limit_invalid() {
    let mut app = TestApp::start().await;

    // Valid requests with invalid power limits
    let resp = app
        .send(
            0,
            SetPowerLimit {
                power_limit_mw: 99_000,
                block: true,
            },
        )
        .await
        .expect("Failed to send request");

    assert_eq!(resp.status(), 400);

    let resp = app
        .send(
            0,
            SetPowerLimit {
                power_limit_mw: 330_000,
                block: true,
            },
        )
        .await
        .expect("Failed to send request");

    assert_eq!(resp.status(), 400);
    let text = resp.text().await.expect("Failed to read response");
    assert!(text.contains("NVML error"));
    assert!(text.contains("invalid"));

    let resp = app
        .send(
            0,
            SetPowerLimit {
                power_limit_mw: 10_000,
                block: false,
            },
        )
        .await
        .expect("Failed to send request");

    // Non-blocking requests should still succeed, but they won't be applied
    assert_eq!(resp.status(), 200);

    assert!(app.power_limit_history_for_gpu(0).is_empty());

    // Invalid request with missing fields
    let client = reqwest::Client::new();
    let url = SetPowerLimit::build_url(&app, 0);
    let resp = client
        .post(url)
        .json(&serde_json::json!(
            {
                "power_limit_wwww": 100_000,
                "block": true
            }
        ))
        .send()
        .await
        .expect("Failed to send request");

    assert_eq!(resp.status(), 400);
    assert!(resp
        .text()
        .await
        .expect("Failed to read response")
        .contains("missing field"));

    for block in [true, false] {
        let url = SetPowerLimit::build_url(&app, 5); // Invalid GPU ID
        let resp = client
            .post(url)
            .json(&serde_json::json!(
                {
                    "power_limit_mw": 100_000,
                    "block": block
                }
            ))
            .send()
            .await
            .expect("Failed to send request");
        assert_eq!(resp.status(), 400);
    }
}

#[tokio::test]
async fn test_set_power_limit_bulk() {
    let mut app = TestApp::start().await;

    let mut set = JoinSet::new();
    for i in 0..10 {
        set.spawn(app.send(
            0,
            SetPowerLimit {
                power_limit_mw: 100_000 + i * 10_000,
                block: false,
            },
        ));
    }
    let mut responses = Vec::with_capacity(10);
    for _ in 0..10 {
        responses.push(set.join_next().await);
    }
    for resp in responses.into_iter() {
        assert_eq!(
            resp.expect("Leaked future")
                .expect("Failed to join future")
                .expect("Failed to send request")
                .status(),
            200
        );
    }

    // After this blocking request finishes, all non-blocking ones should have completed.
    assert_eq!(
        app.send(
            0,
            SetPowerLimit {
                power_limit_mw: 350_000,
                block: true,
            },
        )
        .await
        .expect("Failed to send request")
        .status(),
        400
    );

    let history = app.power_limit_history_for_gpu(0);
    assert_eq!(history.len(), 10);
    assert_eq!(
        HashSet::from_iter(history.into_iter()),
        (0..10)
            .map(|i| 100_000 + i * 10_000)
            .collect::<HashSet<_>>()
    );
}

#[tokio::test]
async fn test_gpu_locked_clocks_single() {
    let mut app = TestApp::start().await;

    let resp = app
        .send(
            0,
            SetGpuLockedClocks {
                min_clock_mhz: 1000,
                max_clock_mhz: 2000,
                block: true,
            },
        )
        .await
        .expect("Failed to send request");

    assert_eq!(resp.status(), 200);
    let history = app.gpu_locked_clocks_history_for_gpu(0);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0], (1000, 2000));
}

#[tokio::test]
async fn test_gpu_locked_clocks_multiple() {
    let mut app = TestApp::start().await;

    let num_requests = 10;
    for i in 0..num_requests {
        let resp = app
            .send(
                i % 4,
                SetGpuLockedClocks {
                    min_clock_mhz: 1000 + i * 100,
                    max_clock_mhz: 2000 + i * 100,
                    block: true,
                },
            )
            .await
            .expect("Failed to send request");

        assert_eq!(resp.status(), 200);
    }

    // Reset the clocks for GPU 2
    assert_eq!(
        app.send(2, ResetGpuLockedClocks { block: true })
            .await
            .expect("Failed to send request")
            .status(),
        200
    );

    assert_eq!(
        app.gpu_locked_clocks_history_for_gpu(0),
        vec![(1000, 2000), (1400, 2400), (1800, 2800)]
    );
    assert_eq!(
        app.gpu_locked_clocks_history_for_gpu(1),
        vec![(1100, 2100), (1500, 2500), (1900, 2900)]
    );
    assert_eq!(
        app.gpu_locked_clocks_history_for_gpu(2),
        vec![(1200, 2200), (1600, 2600), (0, 0)]
    );
    assert_eq!(
        app.gpu_locked_clocks_history_for_gpu(3),
        vec![(1300, 2300), (1700, 2700)]
    );
}

#[tokio::test]
async fn test_gpu_locked_clocks_invalid() {
    let app = TestApp::start().await;

    let client = reqwest::Client::new();
    let url = SetGpuLockedClocks::build_url(&app, 0);
    let resp = client
        .post(url)
        .json(&serde_json::json!(
            {
                "min_clock_mhz": 1000,
                "max_clock_khz": 1100,  // Wrong field name
                "block": true
            }
        ))
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);
    assert!(resp
        .text()
        .await
        .expect("Failed to read response")
        .contains("missing field"));

    let url = SetGpuLockedClocks::build_url(&app, 0);
    let resp = client
        .post(url)
        .json(&serde_json::json!(
            {
                "min_clock_mhz": 1000,
                "max_clock_mhz": "1100",  // Invalid type
                "block": true
            }
        ))
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);
    assert!(resp
        .text()
        .await
        .expect("Failed to read response")
        .contains("invalid type"));

    let url = ResetGpuLockedClocks::build_url(&app, 0);
    let resp = client
        .post(url)
        .json(&serde_json::json!(
            {
                "lego": false  // Wrong field name
            }
        ))
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);
    assert!(resp
        .text()
        .await
        .expect("Failed to read response")
        .contains("missing field"));

    for block in [true, false] {
        let url = ResetGpuLockedClocks::build_url(&app, 5); // Invalid GPU ID
        let resp = client
            .post(url)
            .json(&serde_json::json!(
                {
                    "block": block
                }
            ))
            .send()
            .await
            .expect("Failed to send request");
        assert_eq!(resp.status(), 400);
    }
}

#[tokio::test]
async fn test_gpu_locked_clocks_bulk() {
    let mut app = TestApp::start().await;

    let mut set = JoinSet::new();
    for i in 0..10 {
        set.spawn(app.send(
            0,
            SetGpuLockedClocks {
                min_clock_mhz: 1000 + i * 100,
                max_clock_mhz: 2000 + i * 100,
                block: false,
            },
        ));
    }
    let mut responses = Vec::with_capacity(10);
    for _ in 0..10 {
        responses.push(set.join_next().await);
    }
    for resp in responses.into_iter() {
        assert_eq!(
            resp.expect("Leaked future")
                .expect("Failed to join future")
                .expect("Failed to send request")
                .status(),
            200
        );
    }

    // After this blocking request finishes, all non-blocking ones should have completed.
    assert_eq!(
        app.send(
            0,
            SetGpuLockedClocks {
                min_clock_mhz: 2000,
                max_clock_mhz: 3000,
                block: true,
            },
        )
        .await
        .expect("Failed to send request")
        .status(),
        200
    );

    let history = app.gpu_locked_clocks_history_for_gpu(0);
    assert_eq!(history.len(), 11);
    assert_eq!(
        HashSet::from_iter(history.into_iter()),
        (0..11)
            .map(|i| (1000 + i * 100, 2000 + i * 100))
            .collect::<HashSet<_>>()
    );
}

#[tokio::test]
async fn test_mem_locked_clocks_single() {
    let mut app = TestApp::start().await;

    let resp = app
        .send(
            0,
            SetMemLockedClocks {
                min_clock_mhz: 1000,
                max_clock_mhz: 2000,
                block: true,
            },
        )
        .await
        .expect("Failed to send request");

    assert_eq!(resp.status(), 200);
    let history = app.mem_locked_clocks_history_for_gpu(0);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0], (1000, 2000));
}

#[tokio::test]
async fn test_mem_locked_clocks_multiple() {
    let mut app = TestApp::start().await;

    let num_requests = 10;
    for i in 0..num_requests {
        let resp = app
            .send(
                i % 4,
                SetMemLockedClocks {
                    min_clock_mhz: 1000 + i * 100,
                    max_clock_mhz: 2000 + i * 100,
                    block: true,
                },
            )
            .await
            .expect("Failed to send request");

        assert_eq!(resp.status(), 200);
    }

    // Reset the clocks for GPU 2
    assert_eq!(
        app.send(2, ResetMemLockedClocks { block: true })
            .await
            .expect("Failed to send request")
            .status(),
        200
    );

    assert_eq!(
        app.mem_locked_clocks_history_for_gpu(0),
        vec![(1000, 2000), (1400, 2400), (1800, 2800)]
    );
    assert_eq!(
        app.mem_locked_clocks_history_for_gpu(1),
        vec![(1100, 2100), (1500, 2500), (1900, 2900)]
    );
    assert_eq!(
        app.mem_locked_clocks_history_for_gpu(2),
        vec![(1200, 2200), (1600, 2600), (0, 0)]
    );
    assert_eq!(
        app.mem_locked_clocks_history_for_gpu(3),
        vec![(1300, 2300), (1700, 2700)]
    );
}

#[tokio::test]
async fn test_mem_locked_clocks_invalid() {
    let app = TestApp::start().await;

    let client = reqwest::Client::new();
    let url = SetMemLockedClocks::build_url(&app, 0);
    let resp = client
        .post(url)
        .json(&serde_json::json!(
            {
                "min_clock_mhz": 1000,
                "max_clock_khz": 1100,  // Wrong field name
                "block": true
            }
        ))
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);
    assert!(resp
        .text()
        .await
        .expect("Failed to read response")
        .contains("missing field"));

    let url = SetMemLockedClocks::build_url(&app, 0);
    let resp = client
        .post(url)
        .json(&serde_json::json!(
            {
                "min_clock_mhz": 1000,
                "max_clock_mhz": "1100",  // Invalid type
                "block": true
            }
        ))
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);
    assert!(resp
        .text()
        .await
        .expect("Failed to read response")
        .contains("invalid type"));

    let url = ResetMemLockedClocks::build_url(&app, 0);
    let resp = client
        .post(url)
        .json(&serde_json::json!(
            {
                "lego": false  // Wrong field name
            }
        ))
        .send()
        .await
        .expect("Failed to send request");
    assert_eq!(resp.status(), 400);
    assert!(resp
        .text()
        .await
        .expect("Failed to read response")
        .contains("missing field"));

    for block in [true, false] {
        let url = ResetMemLockedClocks::build_url(&app, 5); // Invalid GPU ID
        let resp = client
            .post(url)
            .json(&serde_json::json!(
                {
                    "block": block
                }
            ))
            .send()
            .await
            .expect("Failed to send request");
        assert_eq!(resp.status(), 400);
    }
}

#[tokio::test]
async fn test_mem_locked_clocks_bulk() {
    let mut app = TestApp::start().await;

    let mut set = JoinSet::new();
    for i in 0..10 {
        set.spawn(app.send(
            0,
            SetMemLockedClocks {
                min_clock_mhz: 1000 + i * 100,
                max_clock_mhz: 2000 + i * 100,
                block: false,
            },
        ));
    }
    let mut responses = Vec::with_capacity(10);
    for _ in 0..10 {
        responses.push(set.join_next().await);
    }
    for resp in responses.into_iter() {
        assert_eq!(
            resp.expect("Leaked future")
                .expect("Failed to join future")
                .expect("Failed to send request")
                .status(),
            200
        );
    }

    // After this blocking request finishes, all non-blocking ones should have completed.
    assert_eq!(
        app.send(
            0,
            SetMemLockedClocks {
                min_clock_mhz: 2000,
                max_clock_mhz: 3000,
                block: true,
            },
        )
        .await
        .expect("Failed to send request")
        .status(),
        200
    );

    let history = app.mem_locked_clocks_history_for_gpu(0);
    assert_eq!(history.len(), 11);
    assert_eq!(
        HashSet::from_iter(history.into_iter()),
        (0..11)
            .map(|i| (1000 + i * 100, 2000 + i * 100))
            .collect::<HashSet<_>>()
    );
}
