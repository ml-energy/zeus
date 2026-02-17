//! Routes for interacting with CPUs

use std::time::Instant;

use actix_web::web::Bytes;
use actix_web::{web, HttpResponse};
use serde::{Deserialize, Serialize};
use tokio::time::Duration;
use tokio_stream::wrappers::WatchStream;
use tokio_stream::StreamExt;

use crate::devices::cpu::power::{CpuPowerBroadcast, CpuPowerSnapshot};
use crate::devices::cpu::{CpuCommand, CpuManagementTasks};
use crate::error::ZeusdError;

#[derive(Serialize, Deserialize, Debug)]
pub struct GetIndexEnergy {
    pub cpu: bool,
    pub dram: bool,
}

impl From<GetIndexEnergy> for CpuCommand {
    fn from(_request: GetIndexEnergy) -> Self {
        CpuCommand::GetIndexEnergy {
            cpu: _request.cpu,
            dram: _request.dram,
        }
    }
}

#[actix_web::post("/{cpu_id}/get_index_energy")]
#[tracing::instrument(
    skip(request, _device_tasks),
    fields(
        cpu_id = %cpu_id,
        cpu = %request.cpu,
        dram = %request.dram,
    )
)]
async fn get_index_energy_handler(
    cpu_id: web::Path<usize>,
    request: web::Json<GetIndexEnergy>,
    _device_tasks: web::Data<CpuManagementTasks>,
) -> Result<HttpResponse, ZeusdError> {
    let now = Instant::now();
    tracing::info!("Received request");
    let cpu_id = cpu_id.into_inner();
    let request = request.into_inner();

    let measurement = _device_tasks
        .send_command_blocking(cpu_id, request.into(), now)
        .await?;

    Ok(HttpResponse::Ok().json(measurement))
}

#[actix_web::get("/{cpu_id}/supports_dram_energy")]
#[tracing::instrument(
    skip(_device_tasks),
    fields(
        cpu_id = %cpu_id,
    )
)]
async fn supports_dram_energy_handler(
    cpu_id: web::Path<usize>,
    _device_tasks: web::Data<CpuManagementTasks>,
) -> Result<HttpResponse, ZeusdError> {
    let now = Instant::now();
    tracing::info!("Received request");
    let cpu_id = cpu_id.into_inner();

    let answer = _device_tasks
        .send_command_blocking(cpu_id, CpuCommand::SupportsDramEnergy, now)
        .await?;

    Ok(HttpResponse::Ok().json(answer))
}

/// Query parameters for CPU power endpoints.
#[derive(Deserialize)]
pub struct CpuPowerQuery {
    /// Comma-separated list of CPU indices. If omitted, all CPUs are included.
    pub cpu_ids: Option<String>,
}

fn parse_cpu_ids(raw: &Option<String>) -> Option<Vec<usize>> {
    raw.as_ref().map(|s| {
        s.split(',')
            .filter_map(|part| part.trim().parse().ok())
            .collect()
    })
}

fn filter_cpu_snapshot(
    snapshot: &CpuPowerSnapshot,
    cpu_ids: &Option<Vec<usize>>,
) -> CpuPowerSnapshot {
    match cpu_ids {
        None => snapshot.clone(),
        Some(ids) => CpuPowerSnapshot {
            timestamp_ms: snapshot.timestamp_ms,
            power_mw: snapshot
                .power_mw
                .iter()
                .filter(|(k, _)| ids.contains(k))
                .map(|(&k, v)| (k, v.clone()))
                .collect(),
        },
    }
}

/// One-shot CPU power reading (computed from RAPL energy deltas).
///
/// Subscribes briefly to wake the poller, waits for a fresh reading (up to
/// 200 ms), then returns the snapshot as JSON.
#[actix_web::get("/power")]
async fn get_cpu_power_handler(
    query: web::Query<CpuPowerQuery>,
    broadcast: web::Data<CpuPowerBroadcast>,
) -> HttpResponse {
    let cpu_ids = parse_cpu_ids(&query.cpu_ids);
    let _guard = broadcast.add_subscriber();
    let mut rx = broadcast.subscribe();
    let _ = tokio::time::timeout(Duration::from_millis(200), rx.changed()).await;
    let snapshot = rx.borrow().clone();
    let filtered = filter_cpu_snapshot(&snapshot, &cpu_ids);
    HttpResponse::Ok().json(filtered)
}

/// SSE stream of CPU power readings.
///
/// The subscriber guard keeps the poller active for the lifetime of the stream.
#[actix_web::get("/power/stream")]
async fn cpu_power_stream_handler(
    query: web::Query<CpuPowerQuery>,
    broadcast: web::Data<CpuPowerBroadcast>,
) -> HttpResponse {
    let guard = broadcast.add_subscriber();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let cpu_ids = parse_cpu_ids(&query.cpu_ids);
    let rx = broadcast.subscribe();
    let stream = WatchStream::new(rx).map(move |snapshot| {
        let _ = &guard;
        let filtered = filter_cpu_snapshot(&snapshot, &cpu_ids);
        let json = serde_json::to_string(&filtered).unwrap_or_default();
        Ok::<_, actix_web::Error>(Bytes::from(format!("data: {json}\n\n")))
    });
    HttpResponse::Ok()
        .insert_header(("Content-Type", "text/event-stream"))
        .insert_header(("Cache-Control", "no-cache"))
        .streaming(stream)
}

pub fn cpu_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(get_index_energy_handler)
        .service(supports_dram_energy_handler)
        .service(get_cpu_power_handler)
        .service(cpu_power_stream_handler);
}
