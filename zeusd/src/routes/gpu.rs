//! Routes for interacting with GPUs

use std::collections::HashMap;
use std::time::Instant;

use actix_web::web::Bytes;
use actix_web::{web, HttpResponse};
use paste::paste;
use serde::{Deserialize, Serialize};
use tokio::time::Duration;
use tokio_stream::wrappers::WatchStream;
use tokio_stream::StreamExt;

use crate::devices::gpu::power::{GpuPowerBroadcast, GpuPowerSnapshot};
use crate::devices::gpu::{GpuCommand, GpuManagementTasks, GpuResponse};
use crate::error::ZeusdError;

/// Query parameters for GPU read endpoints.
/// `gpu_ids` is optional; omit to read all GPUs.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuReadQuery {
    pub gpu_ids: Option<String>,
}

/// Parse a comma-separated list of device indices.
fn parse_gpu_ids(raw: &str) -> Vec<usize> {
    raw.split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect()
}

/// Macro to generate a handler for a GPU command.
///
/// This macro takes
/// - the API name (set_power_limit, set_persistence_mode, etc.),
/// - and a list of `field name: type` pairs of the corresponding `GpuCommand` variant.
///
/// Given this, the macro generates
/// - a query parameter struct named API name (e.g., SetPowerLimit) with `gpu_ids: String`,
///   all the fields specified, and `block: bool`,
/// - an implementation of `From` for the struct to convert it to `GpuCommand`,
/// - a handler function that dispatches the command to each requested GPU.
///
///  Assumptions:
///  - The `GpuCommand` variant name is the same as the API name, but the former is camel case
///    and the latter is snake case (e.g., SetPowerLimit vs. set_power_limit).
macro_rules! impl_handler_for_gpu_command {
    ($api:ident, $path:literal, $($field:ident: $ftype:ty,)*) => {
        paste! {
        // Query parameter structure (includes gpu_ids and block alongside command fields).
        #[derive(Serialize, Deserialize, Debug)]
        #[serde(deny_unknown_fields)]
        pub struct [<$api:camel>] {
            pub gpu_ids: String,
            $(pub $field: $ftype,)*
            pub block: bool,
        }

        // Implement conversion to the GpuCommand variant.
        impl From<[<$api:camel>]> for GpuCommand {
            // Prefixing with underscore to avoid lint errors when $field is empty.
            fn from(_request: [<$api:camel>]) -> Self {
                GpuCommand::[<$api:camel>] {
                    $($field: _request.$field),*
                }
            }
        }

        // Generate the request handler.
        #[actix_web::post($path)]
        #[tracing::instrument(
            skip(query, device_tasks),
            fields(
                gpu_ids = %query.gpu_ids,
                block = %query.block,
                $($field = %query.$field),*
            )
        )]
        async fn [<$api:snake _handler>](
            query: web::Query<[<$api:camel>]>,
            device_tasks: web::Data<GpuManagementTasks>,
        ) -> Result<HttpResponse, ZeusdError> {
            let now = Instant::now();

            tracing::info!("Received request");

            let gpu_ids = parse_gpu_ids(&query.gpu_ids);
            if gpu_ids.is_empty() {
                return Ok(HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "gpu_ids must contain at least one GPU index"
                })));
            }
            let device_count = device_tasks.device_count();
            for &id in &gpu_ids {
                if id >= device_count {
                    return Err(ZeusdError::GpuNotFoundError(id));
                }
            }

            let query = query.into_inner();
            let block = query.block;
            let command: GpuCommand = query.into();

            if block {
                // Execute concurrently across all GPUs and collect results.
                let mut handles = Vec::with_capacity(gpu_ids.len());
                for &gpu_id in &gpu_ids {
                    let cmd = command.clone();
                    let tasks = device_tasks.clone();
                    handles.push(async move {
                        (gpu_id, tasks.send_command_blocking(gpu_id, cmd, now).await)
                    });
                }
                let results = futures::future::join_all(handles).await;
                let mut errors: HashMap<usize, String> = HashMap::new();
                for (gpu_id, result) in results {
                    if let Err(e) = result {
                        errors.insert(gpu_id, e.to_string());
                    }
                }
                if errors.is_empty() {
                    Ok(HttpResponse::Ok().finish())
                } else {
                    Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                        "errors": errors
                    })))
                }
            } else {
                // Non-blocking: send all and collect send results.
                let mut errors: HashMap<usize, String> = HashMap::new();
                for &gpu_id in &gpu_ids {
                    if let Err(e) = device_tasks.send_command_nonblocking(gpu_id, command.clone(), now) {
                        errors.insert(gpu_id, e.to_string());
                    }
                }
                if errors.is_empty() {
                    Ok(HttpResponse::Ok().finish())
                } else {
                    Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                        "errors": errors
                    })))
                }
            }
        }
        }
    };
}

impl_handler_for_gpu_command!(
    set_persistence_mode,
    "/set_persistence_mode",
    enabled: bool,
);

impl_handler_for_gpu_command!(
    set_power_limit,
    "/set_power_limit",
    power_limit_mw: u32,
);

impl_handler_for_gpu_command!(
    set_gpu_locked_clocks,
    "/set_gpu_locked_clocks",
    min_clock_mhz: u32,
    max_clock_mhz: u32,
);

impl_handler_for_gpu_command!(reset_gpu_locked_clocks, "/reset_gpu_locked_clocks",);

impl_handler_for_gpu_command!(
    set_mem_locked_clocks,
    "/set_mem_locked_clocks",
    min_clock_mhz: u32,
    max_clock_mhz: u32,
);

impl_handler_for_gpu_command!(reset_mem_locked_clocks, "/reset_mem_locked_clocks",);

/// Query parameters for the GPU cumulative energy endpoint.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuGetCumulativeEnergyQuery {
    pub gpu_ids: Option<String>,
}

#[derive(Serialize)]
struct GpuEnergyResponse {
    energy_mj: u64,
}

#[actix_web::get("/get_cumulative_energy")]
#[tracing::instrument(skip(query, device_tasks), fields(gpu_ids = ?query.gpu_ids))]
async fn get_cumulative_energy_handler(
    query: web::Query<GpuGetCumulativeEnergyQuery>,
    device_tasks: web::Data<GpuManagementTasks>,
) -> Result<HttpResponse, ZeusdError> {
    let now = Instant::now();
    tracing::info!("Received request");

    let device_count = device_tasks.device_count();
    let gpu_ids: Vec<usize> = match &query.gpu_ids {
        Some(raw) => {
            let ids = parse_gpu_ids(raw);
            if ids.is_empty() {
                return Ok(HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "gpu_ids must contain at least one GPU index"
                })));
            }
            for &id in &ids {
                if id >= device_count {
                    return Err(ZeusdError::GpuNotFoundError(id));
                }
            }
            ids
        }
        None => (0..device_count).collect(),
    };

    let mut handles = Vec::with_capacity(gpu_ids.len());
    for &gpu_id in &gpu_ids {
        let tasks = device_tasks.clone();
        handles.push(async move {
            (
                gpu_id,
                tasks
                    .send_command_blocking(gpu_id, GpuCommand::GetTotalEnergyConsumption, now)
                    .await,
            )
        });
    }
    let results = futures::future::join_all(handles).await;

    let mut response_map: HashMap<String, GpuEnergyResponse> = HashMap::new();
    let mut errors: HashMap<String, String> = HashMap::new();
    for (gpu_id, result) in results {
        match result {
            Ok(GpuResponse::Energy { energy_mj }) => {
                response_map.insert(gpu_id.to_string(), GpuEnergyResponse { energy_mj });
            }
            Ok(_) => {
                errors.insert(gpu_id.to_string(), "Unexpected response type".to_string());
            }
            Err(e) => {
                errors.insert(gpu_id.to_string(), e.to_string());
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

fn filter_snapshot(snapshot: &GpuPowerSnapshot, gpu_ids: &Option<Vec<usize>>) -> GpuPowerSnapshot {
    match gpu_ids {
        None => snapshot.clone(),
        Some(ids) => GpuPowerSnapshot {
            timestamp_ms: snapshot.timestamp_ms,
            power_mw: snapshot
                .power_mw
                .iter()
                .filter(|(k, _)| ids.contains(k))
                .map(|(&k, &v)| (k, v))
                .collect(),
        },
    }
}

/// One-shot GPU power reading.
///
/// Subscribes briefly to wake the poller, waits for a fresh reading (up to
/// 200 ms), then returns the snapshot as JSON. Optionally filtered by
/// `gpu_ids` query parameter (comma-separated GPU indices).
#[actix_web::get("/get_power")]
#[tracing::instrument(skip(broadcast), fields(gpu_ids = ?query.gpu_ids))]
async fn get_power_handler(
    query: web::Query<GpuReadQuery>,
    broadcast: web::Data<GpuPowerBroadcast>,
) -> HttpResponse {
    tracing::info!("Received request");
    let gpu_ids = query.gpu_ids.as_ref().map(|s| parse_gpu_ids(s));
    if let Some(ref ids) = gpu_ids {
        if let Err(unknown) = broadcast.validate_ids(ids) {
            return HttpResponse::BadRequest().json(serde_json::json!({
                "error": format!(
                    "Unknown GPU indices: {:?}. Available: {:?}",
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
    let filtered = filter_snapshot(&snapshot, &gpu_ids);
    HttpResponse::Ok().json(filtered)
}

/// SSE stream of GPU power readings.
///
/// Emits a new event whenever any monitored GPU's power reading changes.
/// The subscriber guard keeps the poller active for the lifetime of the
/// stream. Optionally filtered by `gpu_ids` query parameter (comma-separated
/// GPU indices).
#[actix_web::get("/stream_power")]
#[tracing::instrument(skip(broadcast), fields(gpu_ids = ?query.gpu_ids))]
async fn stream_power_handler(
    query: web::Query<GpuReadQuery>,
    broadcast: web::Data<GpuPowerBroadcast>,
) -> HttpResponse {
    tracing::info!("Received request");
    let gpu_ids = query.gpu_ids.as_ref().map(|s| parse_gpu_ids(s));
    if let Some(ref ids) = gpu_ids {
        if let Err(unknown) = broadcast.validate_ids(ids) {
            return HttpResponse::BadRequest().json(serde_json::json!({
                "error": format!(
                    "Unknown GPU indices: {:?}. Available: {:?}",
                    unknown,
                    broadcast.valid_ids(),
                )
            }));
        }
    }
    let guard = broadcast.add_subscriber();
    // Brief sleep to let the poller produce a first reading.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let rx = broadcast.subscribe();
    let stream = WatchStream::new(rx).map(move |snapshot| {
        let _ = &guard; // prevent drop until stream ends
        let filtered = filter_snapshot(&snapshot, &gpu_ids);
        let json = serde_json::to_string(&filtered).unwrap_or_default();
        Ok::<_, actix_web::Error>(Bytes::from(format!("data: {json}\n\n")))
    });
    HttpResponse::Ok()
        .insert_header(("Content-Type", "text/event-stream"))
        .insert_header(("Cache-Control", "no-cache"))
        .streaming(stream)
}

/// Register GPU routes with the Actix web server.
pub fn gpu_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(set_persistence_mode_handler)
        .service(set_power_limit_handler)
        .service(set_gpu_locked_clocks_handler)
        .service(reset_gpu_locked_clocks_handler)
        .service(set_mem_locked_clocks_handler)
        .service(reset_mem_locked_clocks_handler)
        .service(get_cumulative_energy_handler)
        .service(get_power_handler)
        .service(stream_power_handler);
}
