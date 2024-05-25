use actix_web::{web, HttpResponse};
use tokio::sync::mpsc::channel;

use crate::device::gpu::{GpuCommand, GpuHandlers};

#[derive(serde::Deserialize, Debug)]
struct SetPowerLimitRequest {
    power_limit: usize,
    block: bool,
}

#[actix_web::post("/gpu/{gpu_id}/power_limit")]
pub async fn set_power_limit(
    gpu: web::Path<usize>,
    body: web::Json<SetPowerLimitRequest>,
    gpu_handlers: web::Data<GpuHandlers>,
) -> Result<HttpResponse, actix_web::Error> {
    let gpu = gpu.into_inner();
    let body = body.into_inner();
    let gpu_handlers = gpu_handlers.into_inner();

    println!("{}", gpu);
    println!("{:?}", body);
    println!("{:?}", gpu_handlers);

    if body.block {
        let (tx, mut rx) = channel(1);
        gpu_handlers.send_command(
            gpu,
            GpuCommand::SetPowerLimit {
                power_limit: body.power_limit,
                response: Some(tx),
            },
        );
        rx.recv().await.unwrap()?;
    } else {
        gpu_handlers.send_command(
            gpu,
            GpuCommand::SetPowerLimit {
                power_limit: body.power_limit,
                response: None,
            },
        );
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
