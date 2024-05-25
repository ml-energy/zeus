//! Routes for interacting with GPUs

use actix_web::{web, HttpResponse};

use crate::device::gpu::{GpuCommand, GpuHandlers};

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
    gpu_handlers: web::Data<GpuHandlers>,
) -> Result<HttpResponse, actix_web::Error> {
    tracing::info!("Setting power limit to {} W", request.power_limit / 1000);

    let gpu = gpu.into_inner();
    let request = request.into_inner();
    let gpu_handlers = gpu_handlers.into_inner();

    if request.block {
        gpu_handlers
            .send_command_block(
                gpu,
                GpuCommand::SetPowerLimit {
                    power_limit: request.power_limit,
                },
            )
            .await?;
        tracing::info!("Power limit set successfully");
    } else {
        gpu_handlers.send_command_nonblocking(
            gpu,
            GpuCommand::SetPowerLimit {
                power_limit: request.power_limit,
            },
        )?;
    }
    Ok(HttpResponse::Ok().finish())
}

#[actix_web::post("/gpu/{gpu_id}/frequency")]
pub async fn set_frequency(
    gpu: web::Path<usize>,
    body: web::Json<usize>,
    GpuHandlers: web::Data<GpuHandlers>,
) -> Result<HttpResponse, actix_web::Error> {
    Ok(HttpResponse::Ok().finish())
}
