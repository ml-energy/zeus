//! Server-level routes (discovery, time).

use std::time::{SystemTime, UNIX_EPOCH};

use actix_web::{web, HttpResponse};
use serde::Serialize;

/// Discovery information returned by `GET /discover`.
#[derive(Clone, Debug, Serialize)]
pub struct DiscoveryInfo {
    pub gpu_ids: Vec<usize>,
    pub cpu_ids: Vec<usize>,
    pub dram_available: Vec<bool>,
    pub enabled_api_groups: Vec<String>,
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

/// Register discovery and time routes at the server root.
pub fn server_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(discover_handler).service(time_handler);
}
