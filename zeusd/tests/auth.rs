//! Integration tests for JWT authentication.
//!
//! These tests cover the full behavior matrix:
//!
//! ## Data endpoints (`/gpu/*`, `/cpu/*`)
//!
//! | API group enabled | Auth enabled | Token              | HTTP |
//! |:-----------------:|:------------:|--------------------| :--: |
//! | No                | any          | (any)              | 404  |
//! | Yes               | No           | (any or none)      | 200  |
//! | Yes               | Yes          | None               | 401  |
//! | Yes               | Yes          | Invalid / expired  | 401  |
//! | Yes               | Yes          | Valid, wrong scope  | 403  |
//! | Yes               | Yes          | Valid, correct scope| 200  |
//!
//! ## Always-open endpoints (`/discover`, `/time`)
//!
//! Always return 200, regardless of auth or token.
//!
//! ## `/auth/whoami`
//!
//! | Auth enabled | Token             | HTTP |
//! |:------------:|-------------------|:----:|
//! | No           | (any)             | 404  |
//! | Yes          | None              | 401  |
//! | Yes          | Invalid / expired | 401  |
//! | Yes          | Valid             | 200  |

mod helpers;

use crate::helpers::TestApp;
use zeusd::auth::issue_token;
use zeusd::config::ApiGroup;

const TEST_KEY: &[u8] = b"integration-test-signing-key!!!!";

/// Helper to issue a token with the test key.
fn token(user: &str, scopes: Vec<ApiGroup>, exp: Option<usize>) -> String {
    issue_token(TEST_KEY, user, scopes, exp).unwrap()
}

// =========================================================================
// Always-open endpoints: /discover and /time
// =========================================================================

#[tokio::test]
async fn test_discover_no_auth_server() {
    let app = TestApp::start().await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://127.0.0.1:{}/discover", app.port))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["auth_required"], false);
}

#[tokio::test]
async fn test_discover_auth_server_without_token() {
    let app = TestApp::start_with_auth(TEST_KEY).await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://127.0.0.1:{}/discover", app.port))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["auth_required"], true);
}

#[tokio::test]
async fn test_time_no_auth_server() {
    let app = TestApp::start().await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://127.0.0.1:{}/time", app.port))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn test_time_auth_server_without_token() {
    let app = TestApp::start_with_auth(TEST_KEY).await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://127.0.0.1:{}/time", app.port))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

// =========================================================================
// Disabled API group → unanimous 404 regardless of auth/token
// =========================================================================

#[tokio::test]
async fn test_disabled_group_no_auth_returns_404() {
    // gpu-read not enabled, no auth.
    let app = TestApp::start_with_groups(&[ApiGroup::CpuRead]).await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://127.0.0.1:{}/gpu/get_power", app.port))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_disabled_group_auth_no_token_returns_404() {
    // gpu-read not enabled, auth on, no token → 404 (not 401).
    let app = TestApp::start_with_auth_and_groups(TEST_KEY, &[ApiGroup::CpuRead]).await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://127.0.0.1:{}/gpu/get_power", app.port))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_disabled_group_auth_invalid_token_returns_404() {
    // gpu-read not enabled, auth on, garbage token → 404 (not 401).
    let app = TestApp::start_with_auth_and_groups(TEST_KEY, &[ApiGroup::CpuRead]).await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://127.0.0.1:{}/gpu/get_power", app.port))
        .bearer_auth("not-a-valid-jwt")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_disabled_group_auth_wrong_scope_returns_404() {
    // gpu-read not enabled, auth on, valid token with gpu-read scope → 404 (not 403).
    let app = TestApp::start_with_auth_and_groups(TEST_KEY, &[ApiGroup::CpuRead]).await;
    let client = reqwest::Client::new();
    let t = token("testuser", vec![ApiGroup::GpuRead], Some(9999999999));
    let resp = client
        .get(format!("http://127.0.0.1:{}/gpu/get_power", app.port))
        .bearer_auth(&t)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_disabled_group_auth_correct_scope_returns_404() {
    // gpu-read not enabled, auth on, valid token with gpu-read scope → still 404.
    let app = TestApp::start_with_auth_and_groups(TEST_KEY, &[ApiGroup::CpuRead]).await;
    let client = reqwest::Client::new();
    let t = token("testuser", vec![ApiGroup::GpuRead], Some(9999999999));
    let resp = client
        .get(format!("http://127.0.0.1:{}/gpu/get_power", app.port))
        .bearer_auth(&t)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

// =========================================================================
// Enabled API group, auth disabled → 200 for any/no token
// =========================================================================

#[tokio::test]
async fn test_enabled_group_no_auth_no_token() {
    let app = TestApp::start().await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://127.0.0.1:{}/gpu/get_power", app.port))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn test_enabled_group_no_auth_with_token() {
    // Token is ignored when auth is disabled.
    let app = TestApp::start().await;
    let client = reqwest::Client::new();
    let t = token("testuser", vec![ApiGroup::GpuRead], Some(9999999999));
    let resp = client
        .get(format!("http://127.0.0.1:{}/gpu/get_power", app.port))
        .bearer_auth(&t)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

// =========================================================================
// Enabled API group, auth enabled → depends on token
// =========================================================================

#[tokio::test]
async fn test_enabled_group_auth_no_token_returns_401() {
    let app = TestApp::start_with_auth(TEST_KEY).await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://127.0.0.1:{}/gpu/get_power", app.port))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn test_enabled_group_auth_invalid_token_returns_401() {
    let app = TestApp::start_with_auth(TEST_KEY).await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://127.0.0.1:{}/gpu/get_power", app.port))
        .bearer_auth("not-a-valid-jwt")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn test_enabled_group_auth_expired_token_returns_401() {
    let app = TestApp::start_with_auth(TEST_KEY).await;
    let client = reqwest::Client::new();
    let t = token("testuser", vec![ApiGroup::GpuRead], Some(1000000000));
    let resp = client
        .get(format!("http://127.0.0.1:{}/gpu/get_power", app.port))
        .bearer_auth(&t)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn test_enabled_group_auth_wrong_key_returns_401() {
    let app = TestApp::start_with_auth(TEST_KEY).await;
    let client = reqwest::Client::new();
    let t = issue_token(
        b"wrong-key!!!!!!!!!!!!!!!!!!!!!!!!",
        "testuser",
        vec![ApiGroup::GpuRead],
        Some(9999999999),
    )
    .unwrap();
    let resp = client
        .get(format!("http://127.0.0.1:{}/gpu/get_power", app.port))
        .bearer_auth(&t)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn test_enabled_group_auth_wrong_scope_returns_403() {
    let app = TestApp::start_with_auth(TEST_KEY).await;
    let client = reqwest::Client::new();
    // Token has gpu-read but accessing gpu-control endpoint.
    let t = token("testuser", vec![ApiGroup::GpuRead], Some(9999999999));
    let resp = client
        .post(format!(
            "http://127.0.0.1:{}/gpu/set_power_limit?gpu_ids=0&power_limit_mw=100000&block=true",
            app.port,
        ))
        .bearer_auth(&t)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
}

#[tokio::test]
async fn test_enabled_group_auth_cross_device_scope_returns_403() {
    let app = TestApp::start_with_auth(TEST_KEY).await;
    let client = reqwest::Client::new();
    // Token has gpu-read but accessing cpu-read endpoint.
    let t = token("testuser", vec![ApiGroup::GpuRead], Some(9999999999));
    let resp = client
        .get(format!("http://127.0.0.1:{}/cpu/get_power", app.port))
        .bearer_auth(&t)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403);
}

#[tokio::test]
async fn test_enabled_group_auth_correct_scope_returns_200() {
    let app = TestApp::start_with_auth(TEST_KEY).await;
    let client = reqwest::Client::new();
    let t = token("testuser", vec![ApiGroup::GpuRead], Some(9999999999));
    let resp = client
        .get(format!("http://127.0.0.1:{}/gpu/get_power", app.port))
        .bearer_auth(&t)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn test_enabled_group_auth_correct_scope_gpu_control() {
    let app = TestApp::start_with_auth(TEST_KEY).await;
    let client = reqwest::Client::new();
    let t = token(
        "testuser",
        vec![ApiGroup::GpuControl, ApiGroup::GpuRead],
        Some(9999999999),
    );
    let resp = client
        .post(format!(
            "http://127.0.0.1:{}/gpu/set_power_limit?gpu_ids=0&power_limit_mw=100000&block=true",
            app.port,
        ))
        .bearer_auth(&t)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn test_enabled_group_auth_correct_scope_cpu_read() {
    let app = TestApp::start_with_auth(TEST_KEY).await;
    let client = reqwest::Client::new();
    let t = token("testuser", vec![ApiGroup::CpuRead], Some(9999999999));
    let resp = client
        .get(format!("http://127.0.0.1:{}/cpu/get_power", app.port))
        .bearer_auth(&t)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn test_enabled_group_auth_no_expiry_token() {
    let app = TestApp::start_with_auth(TEST_KEY).await;
    let client = reqwest::Client::new();
    let t = token("testuser", vec![ApiGroup::GpuRead], None);
    let resp = client
        .get(format!("http://127.0.0.1:{}/gpu/get_power", app.port))
        .bearer_auth(&t)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

// =========================================================================
// /auth/whoami
// =========================================================================

#[tokio::test]
async fn test_whoami_no_auth_server_returns_404() {
    let app = TestApp::start().await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://127.0.0.1:{}/auth/whoami", app.port))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_whoami_auth_no_token_returns_401() {
    let app = TestApp::start_with_auth(TEST_KEY).await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://127.0.0.1:{}/auth/whoami", app.port))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn test_whoami_auth_invalid_token_returns_401() {
    let app = TestApp::start_with_auth(TEST_KEY).await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://127.0.0.1:{}/auth/whoami", app.port))
        .bearer_auth("not-a-valid-jwt")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn test_whoami_auth_valid_token() {
    let app = TestApp::start_with_auth(TEST_KEY).await;
    let client = reqwest::Client::new();
    let t = token(
        "alice",
        vec![ApiGroup::GpuRead, ApiGroup::CpuRead],
        Some(9999999999),
    );
    let resp = client
        .get(format!("http://127.0.0.1:{}/auth/whoami", app.port))
        .bearer_auth(&t)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["sub"], "alice");
    let scopes = body["scopes"].as_array().unwrap();
    assert_eq!(scopes.len(), 2);
    assert!(scopes.contains(&serde_json::json!("gpu-read")));
    assert!(scopes.contains(&serde_json::json!("cpu-read")));
    assert_eq!(body["exp"], 9999999999u64);
}

#[tokio::test]
async fn test_whoami_auth_no_expiry_token() {
    let app = TestApp::start_with_auth(TEST_KEY).await;
    let client = reqwest::Client::new();
    let t = token("bob", vec![ApiGroup::GpuControl], None);
    let resp = client
        .get(format!("http://127.0.0.1:{}/auth/whoami", app.port))
        .bearer_auth(&t)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["sub"], "bob");
    assert!(body.get("exp").is_none() || body["exp"].is_null());
}
