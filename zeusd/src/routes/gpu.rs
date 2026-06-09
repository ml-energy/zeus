//! Routes for interacting with GPUs

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use actix_web::{web, HttpResponse};
use paste::paste;
use serde::{Deserialize, Serialize};

use super::{
    parse_device_ids, power_stream_response, resolve_read_device_ids, resolve_stream_device_ids,
};
use crate::devices::gpu::power::{GpuPowerBroadcasts, GpuPowerSnapshot};
use crate::devices::gpu::{GpuCommand, GpuManagementTasks, GpuResponse};
use crate::error::{aggregate_error_response, ZeusdError};
use crate::power_streaming::unix_timestamp_ms;

/// Query parameters for GPU read endpoints.
/// `gpu_ids` is optional; omit to read all GPUs.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuReadQuery {
    pub gpu_ids: Option<String>,
}

/// Macro for a write endpoint that fans a `GpuCommand` out to each
/// requested GPU and returns an empty 200 OK on success.
///
/// Inputs:
/// - `api`: handler name (snake), also the path-tail and the synthesized
///   `<ApiCamel>` query struct + `GpuCommand` variant name.
/// - `path`: route path (e.g. `"/set_power_limit"`).
/// - `field: type, ...`: the variant's command fields. These become
///   query params alongside the always-present `gpu_ids: String` and
///   `block: bool`.
///
/// Assumes the `GpuCommand` variant is `<ApiCamel>` (e.g. `set_power_limit`
/// and `SetPowerLimit`).
macro_rules! impl_write_handler_for_gpu_command {
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

            let gpu_ids = match parse_device_ids(&query.gpu_ids, "GPU") {
                Ok(ids) => ids,
                Err(resp) => return Ok(resp),
            };
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
                let mut errors: HashMap<usize, ZeusdError> = HashMap::new();
                for (gpu_id, result) in results {
                    if let Err(e) = result {
                        errors.insert(gpu_id, e);
                    }
                }
                if errors.is_empty() {
                    Ok(HttpResponse::Ok().finish())
                } else {
                    Ok(aggregate_error_response(errors))
                }
            } else {
                // Non-blocking: send all and collect send results.
                let mut errors: HashMap<usize, ZeusdError> = HashMap::new();
                for &gpu_id in &gpu_ids {
                    if let Err(e) = device_tasks.send_command_nonblocking(gpu_id, command.clone(), now) {
                        errors.insert(gpu_id, e);
                    }
                }
                if errors.is_empty() {
                    Ok(HttpResponse::Ok().finish())
                } else {
                    Ok(aggregate_error_response(errors))
                }
            }
        }
        }
    };
}

impl_write_handler_for_gpu_command!(
    set_persistence_mode,
    "/set_persistence_mode",
    enabled: bool,
);

impl_write_handler_for_gpu_command!(
    set_power_limit,
    "/set_power_limit",
    power_limit_mw: u32,
);

impl_write_handler_for_gpu_command!(
    set_gpu_locked_clocks,
    "/set_gpu_locked_clocks",
    min_clock_mhz: u32,
    max_clock_mhz: u32,
);

impl_write_handler_for_gpu_command!(reset_gpu_locked_clocks, "/reset_gpu_locked_clocks",);

impl_write_handler_for_gpu_command!(
    set_mem_locked_clocks,
    "/set_mem_locked_clocks",
    min_clock_mhz: u32,
    max_clock_mhz: u32,
);

impl_write_handler_for_gpu_command!(reset_mem_locked_clocks, "/reset_mem_locked_clocks",);

/// Macro for a read endpoint that fans a `GpuCommand` out to each requested
/// GPU and returns a JSON map keyed by GPU id.
///
/// Inputs:
/// - `api`: handler name (snake), also the path-tail and the synthesized
///   `<ApiCamel>Response` struct name.
/// - `path`: route path (e.g. `"/get_power_limit"`).
/// - `cmd`: `GpuCommand` variant to dispatch.
/// - `resp`: matching `GpuResponse` variant.
/// - `field: type, ...`: the variant's named fields, in declaration order.
///   The fields become the JSON object body for each GPU.
macro_rules! impl_read_handler_for_gpu_command {
    ($api:ident, $path:literal, $cmd:ident, $resp:ident, $($field:ident: $ftype:ty,)+) => {
        paste! {
            #[derive(Serialize)]
            struct [<$api:camel Response>] {
                $($field: $ftype,)+
            }

            #[actix_web::get($path)]
            #[tracing::instrument(skip(query, device_tasks), fields(gpu_ids = ?query.gpu_ids))]
            async fn [<$api:snake _handler>](
                query: web::Query<GpuReadQuery>,
                device_tasks: web::Data<GpuManagementTasks>,
            ) -> Result<HttpResponse, ZeusdError> {
                let now = Instant::now();

                let gpu_ids = match resolve_read_device_ids(&query.gpu_ids, device_tasks.device_count(), "GPU") {
                    Ok(ids) => ids,
                    Err(resp) => return Ok(resp),
                };

                let mut handles = Vec::with_capacity(gpu_ids.len());
                for &gpu_id in &gpu_ids {
                    let tasks = device_tasks.clone();
                    handles.push(async move {
                        (
                            gpu_id,
                            tasks
                                .send_command_blocking(gpu_id, GpuCommand::$cmd, now)
                                .await,
                        )
                    });
                }
                let results = futures::future::join_all(handles).await;

                let mut response_map: BTreeMap<usize, [<$api:camel Response>]> = BTreeMap::new();
                let mut errors: HashMap<usize, ZeusdError> = HashMap::new();
                for (gpu_id, result) in results {
                    match result {
                        Ok(GpuResponse::$resp { $($field,)+ }) => {
                            response_map.insert(
                                gpu_id,
                                [<$api:camel Response>] { $($field,)+ },
                            );
                        }
                        Ok(_) => {
                            errors.insert(gpu_id, ZeusdError::GpuManagementTaskTerminatedError(gpu_id));
                        }
                        Err(e) => {
                            errors.insert(gpu_id, e);
                        }
                    }
                }

                if errors.is_empty() {
                    Ok(HttpResponse::Ok().json(response_map))
                } else {
                    Ok(aggregate_error_response(errors))
                }
            }
        }
    };
}

impl_read_handler_for_gpu_command!(
    get_cumulative_energy,
    "/get_cumulative_energy",
    GetTotalEnergyConsumption,
    Energy,
    energy_mj: u64,
);

impl_read_handler_for_gpu_command!(
    get_power_limit,
    "/get_power_limit",
    GetPowerLimit,
    PowerLimit,
    power_limit_mw: u32,
);

impl_read_handler_for_gpu_command!(
    get_power_limit_constraints,
    "/get_power_limit_constraints",
    GetPowerLimitConstraints,
    PowerLimitConstraints,
    min_power_limit_mw: u32,
    max_power_limit_mw: u32,
);

impl_read_handler_for_gpu_command!(
    get_persistence_mode,
    "/get_persistence_mode",
    GetPersistenceMode,
    PersistenceMode,
    enabled: bool,
);

/// One-shot GPU power reading.
///
/// Fans out a `GetInstantPower` command to each requested GPU's management
/// task and returns the collected snapshot as JSON.
#[actix_web::get("/get_power")]
#[tracing::instrument(skip(query, device_tasks), fields(gpu_ids = ?query.gpu_ids))]
async fn get_power_handler(
    query: web::Query<GpuReadQuery>,
    device_tasks: web::Data<GpuManagementTasks>,
) -> Result<HttpResponse, ZeusdError> {
    let now = Instant::now();

    let gpu_ids = match resolve_read_device_ids(&query.gpu_ids, device_tasks.device_count(), "GPU")
    {
        Ok(ids) => ids,
        Err(resp) => return Ok(resp),
    };

    let mut handles = Vec::with_capacity(gpu_ids.len());
    for &gpu_id in &gpu_ids {
        let tasks = device_tasks.clone();
        handles.push(async move {
            (
                gpu_id,
                tasks
                    .send_command_blocking(gpu_id, GpuCommand::GetInstantPower, now)
                    .await,
            )
        });
    }
    let results = futures::future::join_all(handles).await;

    let mut power_mw: BTreeMap<usize, u32> = BTreeMap::new();
    let mut errors: HashMap<usize, ZeusdError> = HashMap::new();
    for (gpu_id, result) in results {
        match result {
            Ok(GpuResponse::InstantPower { power_mw: p }) => {
                power_mw.insert(gpu_id, p);
            }
            Ok(_) => {
                errors.insert(gpu_id, ZeusdError::GpuManagementTaskTerminatedError(gpu_id));
            }
            Err(e) => {
                errors.insert(gpu_id, e);
            }
        }
    }

    if !errors.is_empty() {
        return Ok(aggregate_error_response(errors));
    }

    Ok(HttpResponse::Ok().json(GpuPowerSnapshot {
        timestamp_ms: unix_timestamp_ms(),
        power_mw,
    }))
}

/// SSE stream of GPU power readings.
///
/// Emits a new event whenever any monitored GPU's power reading changes.
/// The subscriber guard keeps the poller active for the lifetime of the
/// stream.
#[actix_web::get("/stream_power")]
#[tracing::instrument(skip(query, broadcast), fields(gpu_ids = ?query.gpu_ids))]
async fn stream_power_handler(
    query: web::Query<GpuReadQuery>,
    broadcast: web::Data<GpuPowerBroadcasts>,
) -> HttpResponse {
    match resolve_stream_device_ids(&query.gpu_ids, broadcast.get_ref(), "GPU") {
        Ok(gpu_ids) => power_stream_response(gpu_ids, broadcast.get_ref()),
        Err(response) => response,
    }
}

/// Register read-only GPU monitoring routes.
pub fn gpu_read_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(get_cumulative_energy_handler)
        .service(get_power_handler)
        .service(get_power_limit_handler)
        .service(get_power_limit_constraints_handler)
        .service(get_persistence_mode_handler)
        .service(stream_power_handler);
}

/// Register GPU control (write) routes.
pub fn gpu_control_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(set_persistence_mode_handler)
        .service(set_power_limit_handler)
        .service(set_gpu_locked_clocks_handler)
        .service(reset_gpu_locked_clocks_handler)
        .service(set_mem_locked_clocks_handler)
        .service(reset_mem_locked_clocks_handler);
}
