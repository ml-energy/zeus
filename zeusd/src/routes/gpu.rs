//! Routes for interacting with GPUs

use actix_web::{web, HttpResponse};

use crate::devices::gpu::{GpuCommand, GpuManagementTasks};
use crate::error::ZeusdError;

#[derive(serde::Deserialize, Debug)]
struct SetPowerLimitRequest {
    power_limit: u32,
    block: bool,
}

#[actix_web::post("/gpu/{gpu_id}/power_limit")]
#[tracing::instrument(
    skip(gpu, request, gpu_handlers),
    fields(
        gpu_id = %gpu,
        power_limit = %request.power_limit,
        block = %request.block
    )
)]
pub async fn set_power_limit(
    gpu: web::Path<usize>,
    request: web::Json<SetPowerLimitRequest>,
    gpu_handlers: web::Data<GpuManagementTasks>,
) -> Result<HttpResponse, ZeusdError> {
    tracing::info!("Setting power limit to {} W", request.power_limit / 1000);

    let gpu = gpu.into_inner();
    let request = request.into_inner();
    let gpu_handlers = gpu_handlers.into_inner();

    if request.block {
        gpu_handlers
            .send_command_blocking(
                gpu,
                GpuCommand::SetPowerLimit {
                    power_limit: request.power_limit,
                },
            )
            .await?;
    } else {
        gpu_handlers.send_command_nonblocking(
            gpu,
            GpuCommand::SetPowerLimit {
                power_limit: request.power_limit,
            },
        )?;
    }
    // let response = HttpResponse::Ok().finish();
    // tracing::info!("{:?}", &response);
    // Ok(response)
    Ok(HttpResponse::Ok().finish())
}

#[actix_web::post("/gpu/{gpu_id}/frequency")]
pub async fn set_frequency(
    gpu: web::Path<usize>,
    body: web::Json<usize>,
    gpu_handlers: web::Data<GpuManagementTasks>,
) -> Result<HttpResponse, ZeusdError> {
    Ok(HttpResponse::Ok().finish())
}
