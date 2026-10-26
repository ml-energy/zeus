//! Tracked GPU and CPU power limit routes.

use actix_web::{http::StatusCode, middleware::DefaultHeaders, web, HttpResponse};
use serde::Deserialize;

use crate::devices::executor::QueuePermit;
use crate::devices::{cpu::CpuManagementTasks, gpu::GpuManagementTasks};
use crate::operations::{
    Activation, DeviceType, OperationError, OperationRegistry, OperationSnapshot, RequestedSetting,
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

impl GpuPowerOperation {
    fn validate(&self, count: usize) -> Result<(Vec<usize>, RequestedSetting), OperationError> {
        let ids = ids(&self.gpu_ids, "GPU", count)?;
        if self.power_limit_mw == 0 {
            return Err(OperationError::invalid("power_limit_mw must be positive"));
        }
        Ok((
            ids,
            RequestedSetting {
                power_limit_mw: self.power_limit_mw.into(),
                constraint: None,
            },
        ))
    }
}

impl CpuPowerOperation {
    fn validate(&self, count: usize) -> Result<(Vec<usize>, RequestedSetting), OperationError> {
        let ids = ids(&self.cpu_ids, "CPU", count)?;
        if self.power_limit_mw == 0 || self.power_limit_mw.checked_mul(1000).is_none() {
            return Err(OperationError::invalid(
                "power_limit_mw must be positive and representable in microwatts",
            ));
        }
        if !matches!(
            self.constraint.as_str(),
            "long_term" | "short_term" | "peak_power" | "socket"
        ) {
            return Err(OperationError::invalid(
                "constraint must be long_term, short_term, peak_power, or socket",
            ));
        }
        if self.constraint == "socket" && u32::try_from(self.power_limit_mw).is_err() {
            return Err(OperationError::invalid(
                "power_limit_mw exceeds the HSMP integer range",
            ));
        }
        Ok((
            ids,
            RequestedSetting {
                power_limit_mw: self.power_limit_mw,
                constraint: Some(self.constraint.clone()),
            },
        ))
    }
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
    let (ids, requested) = query.validate(tasks.device_count())?;
    let permits = ids
        .iter()
        .map(|&id| tasks.reserve_operation(id))
        .collect::<Result<Vec<_>, _>>()?;
    let prepared = operations.prepare(DeviceType::Gpu, &ids, requested, query.start_within_ms)?;
    let activation = operations.activate(&prepared.id, DeviceType::Gpu, |_| Ok(permits))?;
    Ok(accepted(dispatch_gpu(activation, &tasks)?))
}

async fn submit_cpu(
    query: web::Query<CpuPowerOperation>,
    tasks: web::Data<CpuManagementTasks>,
    operations: web::Data<OperationRegistry>,
) -> Result<HttpResponse, OperationError> {
    let (ids, requested) = query.validate(tasks.device_count())?;
    let permits = ids
        .iter()
        .map(|&id| tasks.reserve_operation(id))
        .collect::<Result<Vec<_>, _>>()?;
    let prepared = operations.prepare(DeviceType::Cpu, &ids, requested, query.start_within_ms)?;
    let activation = operations.activate(&prepared.id, DeviceType::Cpu, |_| Ok(permits))?;
    Ok(accepted(dispatch_cpu(activation, &tasks)?))
}

async fn prepare_gpu(
    query: web::Query<GpuPowerOperation>,
    tasks: web::Data<GpuManagementTasks>,
    operations: web::Data<OperationRegistry>,
) -> Result<HttpResponse, OperationError> {
    let (ids, requested) = query.validate(tasks.device_count())?;
    Ok(accepted(operations.prepare(
        DeviceType::Gpu,
        &ids,
        requested,
        query.start_within_ms,
    )?))
}

async fn prepare_cpu(
    query: web::Query<CpuPowerOperation>,
    tasks: web::Data<CpuManagementTasks>,
    operations: web::Data<OperationRegistry>,
) -> Result<HttpResponse, OperationError> {
    let (ids, requested) = query.validate(tasks.device_count())?;
    Ok(accepted(operations.prepare(
        DeviceType::Cpu,
        &ids,
        requested,
        query.start_within_ms,
    )?))
}

async fn activate_gpu(
    id: web::Path<String>,
    tasks: web::Data<GpuManagementTasks>,
    operations: web::Data<OperationRegistry>,
) -> Result<HttpResponse, OperationError> {
    let activation = operations.activate(&id, DeviceType::Gpu, |snapshot| {
        snapshot
            .devices
            .keys()
            .map(|&id| tasks.reserve_operation(id))
            .collect::<Result<Vec<_>, _>>()
    })?;
    Ok(accepted(dispatch_gpu(activation, &tasks)?))
}

fn dispatch_gpu(
    activation: Activation<Vec<QueuePermit>>,
    tasks: &GpuManagementTasks,
) -> Result<OperationSnapshot, OperationError> {
    let snapshot = match activation {
        Activation::Existing(snapshot) => snapshot,
        Activation::Dispatch {
            snapshot,
            tickets,
            reservation,
        } => {
            let power_limit_mw = u32::try_from(snapshot.requested.power_limit_mw)
                .map_err(|_| OperationError::invalid("Stored GPU power limit exceeds u32"))?;
            for ((&id, ticket), permit) in snapshot.devices.keys().zip(tickets).zip(reservation) {
                tasks.send_operation(id, power_limit_mw, ticket, permit);
            }
            snapshot
        }
    };
    Ok(snapshot)
}

async fn activate_cpu(
    id: web::Path<String>,
    tasks: web::Data<CpuManagementTasks>,
    operations: web::Data<OperationRegistry>,
) -> Result<HttpResponse, OperationError> {
    let activation = operations.activate(&id, DeviceType::Cpu, |snapshot| {
        snapshot
            .devices
            .keys()
            .map(|&id| tasks.reserve_operation(id))
            .collect::<Result<Vec<_>, _>>()
    })?;
    Ok(accepted(dispatch_cpu(activation, &tasks)?))
}

fn dispatch_cpu(
    activation: Activation<Vec<QueuePermit>>,
    tasks: &CpuManagementTasks,
) -> Result<OperationSnapshot, OperationError> {
    let snapshot = match activation {
        Activation::Existing(snapshot) => snapshot,
        Activation::Dispatch {
            snapshot,
            tickets,
            reservation,
        } => {
            let constraint =
                snapshot.requested.constraint.as_ref().ok_or_else(|| {
                    OperationError::invalid("Stored CPU operation has no constraint")
                })?;
            for ((&id, ticket), permit) in snapshot.devices.keys().zip(tickets).zip(reservation) {
                tasks.send_operation(
                    id,
                    constraint.clone(),
                    snapshot.requested.power_limit_mw,
                    ticket,
                    permit,
                );
            }
            snapshot
        }
    };
    Ok(snapshot)
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
    ($name:ident, $device:ident, $submit:ident, $prepare:ident, $activate:ident) => {
        pub fn $name(cfg: &mut web::ServiceConfig) {
            cfg.service(
                web::scope("/set_power_limit/operations")
                    .wrap(DefaultHeaders::new().add(("Cache-Control", "no-store")))
                    .app_data(web::Data::new(DeviceType::$device))
                    .app_data(web::QueryConfig::default().error_handler(|error, _| {
                        OperationError::invalid(error.to_string()).into()
                    }))
                    .route("/submit", web::post().to($submit))
                    .route("/prepare", web::post().to($prepare))
                    .route("/{id}/activate", web::post().to($activate))
                    .route("/{id}", web::get().to(status))
                    .route("/{id}", web::delete().to(cancel))
                    .default_service(web::route().to(|| async {
                        HttpResponse::build(StatusCode::METHOD_NOT_ALLOWED).finish()
                    })),
            );
        }
    };
}

routes!(
    gpu_operation_routes,
    Gpu,
    submit_gpu,
    prepare_gpu,
    activate_gpu
);
routes!(
    cpu_operation_routes,
    Cpu,
    submit_cpu,
    prepare_cpu,
    activate_cpu
);
