//! Routes for interacting with CPUs

use std::collections::HashMap;
use std::time::Instant;

use actix_web::web::Bytes;
use actix_web::{web, HttpResponse};
use serde::{Deserialize, Serialize};
use tokio::time::Duration;
use tokio_stream::wrappers::WatchStream;
use tokio_stream::StreamExt;

use crate::devices::cpu::power::{CpuPowerBroadcast, CpuPowerSnapshot};
use crate::devices::cpu::{CpuCommand, CpuManagementTasks, RaplResponse};
use crate::error::ZeusdError;

/// Query parameters for CPU read endpoints.
/// `cpu_ids` is optional; omit to read all CPUs.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpuReadQuery {
    pub cpu_ids: Option<String>,
}

/// Parse a comma-separated list of device indices.
fn parse_cpu_ids(raw: &str) -> Vec<usize> {
    raw.split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect()
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct GetCumulativeEnergy {
    pub cpu_ids: String,
    pub cpu: bool,
    pub dram: bool,
}

impl From<GetCumulativeEnergy> for CpuCommand {
    fn from(request: GetCumulativeEnergy) -> Self {
        CpuCommand::GetIndexEnergy {
            cpu: request.cpu,
            dram: request.dram,
        }
    }
}

#[actix_web::get("/get_cumulative_energy")]
#[tracing::instrument(
    skip(query, device_tasks),
    fields(
        cpu_ids = %query.cpu_ids,
        cpu = %query.cpu,
        dram = %query.dram,
    )
)]
async fn get_cumulative_energy_handler(
    query: web::Query<GetCumulativeEnergy>,
    device_tasks: web::Data<CpuManagementTasks>,
) -> Result<HttpResponse, ZeusdError> {
    let now = Instant::now();
    tracing::info!("Received request");

    let cpu_ids = parse_cpu_ids(&query.cpu_ids);
    if cpu_ids.is_empty() {
        return Ok(HttpResponse::BadRequest().json(serde_json::json!({
            "error": "cpu_ids must contain at least one CPU index"
        })));
    }
    let device_count = device_tasks.device_count();
    for &id in &cpu_ids {
        if id >= device_count {
            return Err(ZeusdError::CpuNotFoundError(id));
        }
    }

    // Execute concurrently for all requested CPUs.
    let mut handles = Vec::with_capacity(cpu_ids.len());
    for &cpu_id in &cpu_ids {
        let cmd: CpuCommand = CpuCommand::GetIndexEnergy {
            cpu: query.cpu,
            dram: query.dram,
        };
        let tasks = device_tasks.clone();
        handles.push(async move { (cpu_id, tasks.send_command_blocking(cpu_id, cmd, now).await) });
    }
    let results = futures::future::join_all(handles).await;

    let mut response_map: HashMap<String, RaplResponse> = HashMap::new();
    let mut errors: HashMap<String, String> = HashMap::new();
    for (cpu_id, result) in results {
        match result {
            Ok(measurement) => {
                response_map.insert(cpu_id.to_string(), measurement);
            }
            Err(e) => {
                errors.insert(cpu_id.to_string(), e.to_string());
            }
        }
    }

    if errors.is_empty() {
        Ok(HttpResponse::Ok().json(response_map))
    } else {
        Ok(HttpResponse::InternalServerError().json(serde_json::json!({
            "errors": errors
        })))
    }
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
#[actix_web::get("/get_power")]
#[tracing::instrument(skip(broadcast), fields(cpu_ids = ?query.cpu_ids))]
async fn get_cpu_power_handler(
    query: web::Query<CpuReadQuery>,
    broadcast: web::Data<CpuPowerBroadcast>,
) -> HttpResponse {
    tracing::info!("Received request");
    let cpu_ids = query.cpu_ids.as_ref().map(|s| parse_cpu_ids(s));
    if let Some(ref ids) = cpu_ids {
        if let Err(unknown) = broadcast.validate_ids(ids) {
            return HttpResponse::BadRequest().json(serde_json::json!({
                "error": format!(
                    "Unknown CPU indices: {:?}. Available: {:?}",
                    unknown,
                    broadcast.valid_ids(),
                )
            }));
        }
    }
    let _guard = broadcast.add_subscriber();
    let mut rx = broadcast.subscribe();
    rx.borrow_and_update();
    let _ = tokio::time::timeout(Duration::from_millis(200), rx.changed()).await;
    let snapshot = rx.borrow().clone();
    let filtered = filter_cpu_snapshot(&snapshot, &cpu_ids);
    HttpResponse::Ok().json(filtered)
}

/// SSE stream of CPU power readings.
///
/// The subscriber guard keeps the poller active for the lifetime of the stream.
#[actix_web::get("/stream_power")]
#[tracing::instrument(skip(broadcast), fields(cpu_ids = ?query.cpu_ids))]
async fn cpu_power_stream_handler(
    query: web::Query<CpuReadQuery>,
    broadcast: web::Data<CpuPowerBroadcast>,
) -> HttpResponse {
    tracing::info!("Received request");
    let cpu_ids = query.cpu_ids.as_ref().map(|s| parse_cpu_ids(s));
    if let Some(ref ids) = cpu_ids {
        if let Err(unknown) = broadcast.validate_ids(ids) {
            return HttpResponse::BadRequest().json(serde_json::json!({
                "error": format!(
                    "Unknown CPU indices: {:?}. Available: {:?}",
                    unknown,
                    broadcast.valid_ids(),
                )
            }));
        }
    }
    let guard = broadcast.add_subscriber();
    tokio::time::sleep(Duration::from_millis(100)).await;
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
    cfg.service(get_cumulative_energy_handler)
        .service(get_cpu_power_handler)
        .service(cpu_power_stream_handler);
}
