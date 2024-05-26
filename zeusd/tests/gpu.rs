mod helpers;

use crate::helpers::TestApp;

#[tokio::test]
async fn test_set_persistent_mode() {
    let mut app = TestApp::start().await;

    let resp = app.set_persistent_mode(0, true, true).await;

    assert_eq!(resp.status(), 200);
    assert_eq!(
        app.observers[0].persistent_mode_rx.recv().await.unwrap(),
        true
    );
}
