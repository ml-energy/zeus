mod helpers;

use zeusd::routes::gpu::SetPersistentMode;

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
}
