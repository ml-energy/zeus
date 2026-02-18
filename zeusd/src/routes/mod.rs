//! Routes and handlers for interacting with devices

pub mod cpu;
pub mod gpu;

pub use cpu::cpu_routes;
pub use gpu::gpu_routes;

use actix_web::{web, HttpResponse};
use serde::Serialize;

/// Discovery information returned by `GET /discover`.
#[derive(Clone, Serialize)]
pub struct DiscoveryInfo {
    pub gpu_ids: Vec<usize>,
    pub cpu_ids: Vec<usize>,
    pub dram_available: Vec<bool>,
}

#[actix_web::get("/discover")]
async fn discover_handler(info: web::Data<DiscoveryInfo>) -> HttpResponse {
    HttpResponse::Ok().json(info.as_ref())
}

/// Register the discovery route at the server root.
pub fn discovery_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(discover_handler);
}
