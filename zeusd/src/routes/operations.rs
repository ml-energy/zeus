//! Tracked GPU and CPU power limit routes.

use actix_web::{http::StatusCode, middleware::DefaultHeaders, web, HttpResponse};
use serde::Deserialize;

use crate::devices::{cpu::CpuManagementTasks, gpu::GpuManagementTasks};
use crate::operations::{
    DeviceType, OperationError, OperationRegistry, OperationSnapshot, RequestedSetting,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GpuPowerOperation {
    gpu_ids: String,
    power_limit_mw: u32,
    start_within_ms: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CpuPowerOperation {
    cpu_ids: String,
    constraint: String,
    power_limit_mw: u64,
    start_within_ms: u64,
}

fn ids(raw: &str, device: &str, count: usize) -> Result<Vec<usize>, OperationError> {
    let ids = super::parse_device_ids(raw, device).map_err(|_| {
        OperationError::invalid(format!(
            "{device} IDs must be a nonempty comma-separated list of indices"
        ))
    })?;
    if ids.len() != raw.split(',').count() {
        return Err(OperationError::invalid(
            "Duplicate device indices are not allowed",
        ));
    }
    for &id in &ids {
        if id >= count {
            return Err(OperationError::invalid(format!(
                "{device} {id} does not exist"
            )));
        }
    }
    Ok(ids)
}

fn accepted(snapshot: OperationSnapshot) -> HttpResponse {
    HttpResponse::Accepted()
        .insert_header(("Cache-Control", "no-store"))
        .insert_header(("Location", snapshot.status_url.clone()))
        .json(snapshot)
}

async fn submit_gpu(
    query: web::Query<GpuPowerOperation>,
    tasks: web::Data<GpuManagementTasks>,
    operations: web::Data<OperationRegistry>,
) -> Result<HttpResponse, OperationError> {
    let ids = ids(&query.gpu_ids, "GPU", tasks.device_count())?;
    if query.power_limit_mw == 0 {
        return Err(OperationError::invalid("power_limit_mw must be positive"));
    }
    let permits = ids
        .iter()
        .map(|&id| tasks.reserve_operation(id))
        .collect::<Result<Vec<_>, _>>()?;
    let (snapshot, tickets) = operations.admit(
        DeviceType::Gpu,
        &ids,
        RequestedSetting {
            power_limit_mw: query.power_limit_mw.into(),
            constraint: None,
        },
        query.start_within_ms,
    )?;
    for ((id, ticket), permit) in ids.into_iter().zip(tickets).zip(permits) {
        tasks.send_operation(id, query.power_limit_mw, ticket, permit);
    }
    Ok(accepted(snapshot))
}

async fn submit_cpu(
    query: web::Query<CpuPowerOperation>,
    tasks: web::Data<CpuManagementTasks>,
    operations: web::Data<OperationRegistry>,
) -> Result<HttpResponse, OperationError> {
    let ids = ids(&query.cpu_ids, "CPU", tasks.device_count())?;
    if query.power_limit_mw == 0 || query.power_limit_mw.checked_mul(1000).is_none() {
        return Err(OperationError::invalid(
            "power_limit_mw must be positive and representable in microwatts",
        ));
    }
    if !matches!(
        query.constraint.as_str(),
        "long_term" | "short_term" | "peak_power" | "socket"
    ) {
        return Err(OperationError::invalid(
            "constraint must be long_term, short_term, peak_power, or socket",
        ));
    }
    if query.constraint == "socket" && u32::try_from(query.power_limit_mw).is_err() {
        return Err(OperationError::invalid(
            "power_limit_mw exceeds the HSMP integer range",
        ));
    }
    let permits = ids
        .iter()
        .map(|&id| tasks.reserve_operation(id))
        .collect::<Result<Vec<_>, _>>()?;
    let (snapshot, tickets) = operations.admit(
        DeviceType::Cpu,
        &ids,
        RequestedSetting {
            power_limit_mw: query.power_limit_mw,
            constraint: Some(query.constraint.clone()),
        },
        query.start_within_ms,
    )?;
    for ((id, ticket), permit) in ids.into_iter().zip(tickets).zip(permits) {
        tasks.send_operation(
            id,
            query.constraint.clone(),
            query.power_limit_mw,
            ticket,
            permit,
        );
    }
    Ok(accepted(snapshot))
}

async fn status(
    id: web::Path<String>,
    operations: web::Data<OperationRegistry>,
    device: web::Data<DeviceType>,
) -> Result<HttpResponse, OperationError> {
    Ok(HttpResponse::Ok().json(operations.get(&id, **device)?))
}

async fn cancel(
    id: web::Path<String>,
    operations: web::Data<OperationRegistry>,
    device: web::Data<DeviceType>,
) -> Result<HttpResponse, OperationError> {
    Ok(HttpResponse::Ok().json(operations.cancel(&id, **device)?))
}

macro_rules! routes {
    ($name:ident, $device:ident, $submit:ident) => {
        pub(super) fn $name(cfg: &mut web::ServiceConfig) {
            cfg.service(
                web::scope("/set_power_limit/operations")
                    .wrap(DefaultHeaders::new().add(("Cache-Control", "no-store")))
                    .app_data(web::Data::new(DeviceType::$device))
                    .app_data(web::QueryConfig::default().error_handler(|error, _| {
                        OperationError::invalid(error.to_string()).into()
                    }))
                    .route("", web::post().to($submit))
                    .route("/{id}", web::get().to(status))
                    .route("/{id}", web::delete().to(cancel))
                    .default_service(web::route().to(|| async {
                        HttpResponse::build(StatusCode::METHOD_NOT_ALLOWED).finish()
                    })),
            );
        }
    };
}

routes!(gpu_operation_routes, Gpu, submit_gpu);
routes!(cpu_operation_routes, Cpu, submit_cpu);
