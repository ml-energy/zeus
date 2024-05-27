mod helpers;

use zeusd::routes::gpu::{SetPersistentMode, SetPowerLimit};

use crate::helpers::TestApp;

#[tokio::test]
async fn test_set_persistent_mode() {
    let mut app = TestApp::start().await;

    let resp = app
        .send(
            0,
            SetPersistentMode {
                enabled: true,
                block: true,
            },
        )
        .await;

    assert_eq!(resp.status(), 200);
    let history = app.persistent_mode_history_for_gpu(0);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0], true);

    let num_requests = 10;
    for i in 0..num_requests {
        let resp = app
            .send(
                i % 4,
                SetPersistentMode {
                    enabled: (i / 4) % 2 == 0,
                    block: true,
                },
            )
            .await;

        assert_eq!(resp.status(), 200);
    }

    assert_eq!(
        app.persistent_mode_history_for_gpu(0),
        vec![true, false, true]
    );
    assert_eq!(
        app.persistent_mode_history_for_gpu(1),
        vec![true, false, true]
    );
    assert_eq!(app.persistent_mode_history_for_gpu(2), vec![true, false]);
    assert_eq!(app.persistent_mode_history_for_gpu(3), vec![true, false]);
}

#[tokio::test]
async fn test_set_power_limit() {
    let mut app = TestApp::start().await;

    let resp = app
        .send(
            0,
            SetPowerLimit {
                power_limit_mw: 100_000,
                block: true,
            },
        )
        .await;

    assert_eq!(resp.status(), 200);
    let history = app.power_limit_history_for_gpu(0);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0], 100_000);

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
            .await;

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

    let resp = app
        .send(
            0,
            SetPowerLimit {
                power_limit_mw: 99_000,
                block: true,
            },
        )
        .await;
    assert_eq!(resp.status(), 400);

    let resp = app
        .send(
            0,
            SetPowerLimit {
                power_limit_mw: 330_000,
                block: true,
            },
        )
        .await;
    assert_eq!(resp.status(), 400);
}
