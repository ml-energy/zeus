//! Routes for interacting with CPUs

use actix_web::{web, HttpResponse};
use serde::{Deserialize, Serialize};
use std::time::Instant;

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

pub fn cpu_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(get_index_energy_handler);
    cfg.service(supports_dram_energy_handler);
}
