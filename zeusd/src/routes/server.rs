//! Server-level routes (discovery, time, whoami).

use std::time::{SystemTime, UNIX_EPOCH};

use actix_web::{web, HttpMessage, HttpRequest, HttpResponse};
use serde::Serialize;

use crate::auth::Claims;
use crate::config::ApiGroup;

/// Discovery information returned by `GET /discover`.
#[derive(Clone, Debug, Serialize)]
pub struct DiscoveryInfo {
    pub gpu_ids: Vec<usize>,
    pub cpu_ids: Vec<usize>,
    pub dram_available: Vec<bool>,
    pub enabled_api_groups: Vec<String>,
    pub auth_required: bool,
}

#[actix_web::get("/discover")]
async fn discover_handler(info: web::Data<DiscoveryInfo>) -> HttpResponse {
    HttpResponse::Ok().json(info.as_ref())
}

/// Response for `GET /time`.
#[derive(Serialize)]
struct TimeResponse {
    timestamp_ms: u64,
}

/// Return the daemon's current Unix timestamp in milliseconds.
#[actix_web::get("/time")]
async fn time_handler() -> HttpResponse {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    HttpResponse::Ok().json(TimeResponse { timestamp_ms })
}

/// Response for `GET /auth/whoami`.
#[derive(Serialize)]
struct WhoamiResponse {
    sub: String,
    scopes: Vec<ApiGroup>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exp: Option<usize>,
}

/// Return the authenticated user's identity and scopes.
///
/// This endpoint requires a valid token (the auth middleware enforces
/// this). It reads the `Claims` inserted by the middleware and echoes
/// them back, allowing clients to verify their token and check which
/// scopes they have.
#[actix_web::get("/auth/whoami")]
async fn whoami_handler(req: HttpRequest) -> HttpResponse {
    let extensions = req.extensions();
    // The auth middleware inserts Claims for all authenticated requests
    // and returns 404 for /auth/* when auth is disabled, so Claims
    // should always be present here.
    match extensions.get::<Claims>() {
        Some(claims) => HttpResponse::Ok().json(WhoamiResponse {
            sub: claims.sub.clone(),
            scopes: claims.scopes.clone(),
            exp: claims.exp,
        }),
        None => HttpResponse::InternalServerError().json(serde_json::json!({
            "error": "Claims not found in request extensions."
        })),
    }
}

/// Register discovery, time, and auth routes at the server root.
pub fn server_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(discover_handler)
        .service(time_handler)
        .service(whoami_handler);
}
