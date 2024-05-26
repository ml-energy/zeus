//! Routes for interacting with GPUs

use actix_web::{web, HttpResponse};

use crate::devices::gpu::{GpuCommand, GpuManagementTasks};
use crate::error::ZeusdError;

#[derive(serde::Deserialize, Debug)]
#[cfg_attr(feature = "testing", derive(serde::Serialize))]
pub struct SetPersistentModeRequest {
    pub enabled: bool,
    pub block: bool,
}

impl From<SetPersistentModeRequest> for GpuCommand {
    fn from(request: SetPersistentModeRequest) -> Self {
        GpuCommand::SetPersistentMode {
            enabled: request.enabled,
        }
    }
}

#[actix_web::post("/{gpu_id}/persistent_mode")]
#[tracing::instrument(
    skip(gpu, request, device_tasks),
    fields(
        gpu_id = %gpu,
        enabled = %request.enabled,
        block = %request.block
    )
)]
pub async fn set_persistent_mode_handler(
    gpu: web::Path<usize>,
    request: web::Json<SetPersistentModeRequest>,
    device_tasks: web::Data<GpuManagementTasks>,
) -> Result<HttpResponse, ZeusdError> {
    let gpu = gpu.into_inner();
    let request = request.into_inner();

    tracing::info!(
        "Received reqeust to set GPU {}'s persistent mode to {} W",
        gpu,
        if request.enabled {
            "enabled"
        } else {
            "disabled"
        },
    );

    if request.block {
        device_tasks
            .send_command_blocking(gpu, request.into())
            .await?;
    } else {
        device_tasks.send_command_nonblocking(gpu, request.into())?;
    }

    Ok(HttpResponse::Ok().finish())
}

#[derive(serde::Deserialize, Debug)]
struct SetPowerLimitRequest {
    power_limit_uw: u32,
    block: bool,
}

impl From<SetPowerLimitRequest> for GpuCommand {
    fn from(request: SetPowerLimitRequest) -> Self {
        GpuCommand::SetPowerLimit {
            power_limit_mw: request.power_limit_uw,
        }
    }
}

#[actix_web::post("/{gpu_id}/power_limit")]
#[tracing::instrument(
    skip(gpu, request, device_tasks),
    fields(
        gpu_id = %gpu,
        power_limit = %request.power_limit_uw,
        block = %request.block
    )
)]
pub async fn set_power_limit_handler(
    gpu: web::Path<usize>,
    request: web::Json<SetPowerLimitRequest>,
    device_tasks: web::Data<GpuManagementTasks>,
) -> Result<HttpResponse, ZeusdError> {
    let gpu = gpu.into_inner();
    let request = request.into_inner();

    tracing::info!(
        "Received reqeust to set GPU {}'s power limit to {} W",
        gpu,
        request.power_limit_uw / 1000,
    );

    if request.block {
        device_tasks
            .send_command_blocking(gpu, request.into())
            .await?;
    } else {
        device_tasks.send_command_nonblocking(gpu, request.into())?;
    }

    Ok(HttpResponse::Ok().finish())
}

#[derive(serde::Deserialize, Debug)]
struct SetGpuLockedClocksRequest {
    min_clock_mhz: u32,
    max_clock_mhz: u32,
    block: bool,
}

impl From<SetGpuLockedClocksRequest> for GpuCommand {
    fn from(request: SetGpuLockedClocksRequest) -> Self {
        GpuCommand::SetGpuLockedClocks {
            min_clock_mhz: request.min_clock_mhz,
            max_clock_mhz: request.max_clock_mhz,
        }
    }
}

#[actix_web::post("/{gpu_id}/gpu_locked_clocks")]
#[tracing::instrument(
    skip(gpu, request, device_tasks),
    fields(
        gpu_id = %gpu,
        min_clock_mhz = %request.min_clock_mhz,
        max_clock_mhz = %request.max_clock_mhz,
        block = %request.block
    )
)]
pub async fn set_gpu_locked_clocks_handler(
    gpu: web::Path<usize>,
    request: web::Json<SetGpuLockedClocksRequest>,
    device_tasks: web::Data<GpuManagementTasks>,
) -> Result<HttpResponse, ZeusdError> {
    let gpu = gpu.into_inner();
    let request = request.into_inner();

    tracing::info!(
        "Received reqeust to set GPU {}'s gpu locked clocks to [{}, {}] MHz",
        gpu,
        request.min_clock_mhz,
        request.max_clock_mhz,
    );

    if request.block {
        device_tasks
            .send_command_blocking(gpu, request.into())
            .await?;
    } else {
        device_tasks.send_command_nonblocking(gpu, request.into())?;
    }

    Ok(HttpResponse::Ok().finish())
}

#[derive(serde::Deserialize, Debug)]
struct SetMemLockedClocksRequest {
    min_clock_mhz: u32,
    max_clock_mhz: u32,
    block: bool,
}

impl From<SetMemLockedClocksRequest> for GpuCommand {
    fn from(request: SetMemLockedClocksRequest) -> Self {
        GpuCommand::SetMemLockedClocks {
            min_clock_mhz: request.min_clock_mhz,
            max_clock_mhz: request.max_clock_mhz,
        }
    }
}

#[actix_web::post("/{gpu_id}/mem_locked_clocks")]
#[tracing::instrument(
    skip(gpu, request, device_tasks),
    fields(
        gpu_id = %gpu,
        min_clock_mhz = %request.min_clock_mhz,
        max_clock_mhz = %request.max_clock_mhz,
        block = %request.block
    )
)]
pub async fn set_mem_locked_clocks_handler(
    gpu: web::Path<usize>,
    request: web::Json<SetMemLockedClocksRequest>,
    device_tasks: web::Data<GpuManagementTasks>,
) -> Result<HttpResponse, ZeusdError> {
    let gpu = gpu.into_inner();
    let request = request.into_inner();

    tracing::info!(
        "Received reqeust to set GPU {}'s memory locked clocks to [{}, {}] MHz",
        gpu,
        request.min_clock_mhz,
        request.max_clock_mhz,
    );

    if request.block {
        device_tasks
            .send_command_blocking(gpu, request.into())
            .await?;
    } else {
        device_tasks.send_command_nonblocking(gpu, request.into())?;
    }

    Ok(HttpResponse::Ok().finish())
}
