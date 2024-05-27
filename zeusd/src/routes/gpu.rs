//! Routes for interacting with GPUs

use actix_web::{web, HttpResponse};
use paste::paste;
use serde::{Deserialize, Serialize};

use crate::devices::gpu::{GpuCommand, GpuManagementTasks};
use crate::error::ZeusdError;

macro_rules! impl_handler_for_gpu_command {
    ($action:ident, $api:ident, $path:expr, $($field:ident),*) => {
        // Implement conversion to the GpuCommand variant.
        paste! {
        impl From<[<$action:camel $api:camel>]> for GpuCommand {
            // Prefixing with underscore to avoid lint errors when $field is empty.
            fn from(_request: [<$action:camel $api:camel>]) -> Self {
                GpuCommand::[<$action:camel $api:camel>] {
                    $($field: _request.$field),*
                }
            }
        }

        // Generate the request handler.
        #[actix_web::post($path)]
        #[tracing::instrument(
            skip(gpu, request, device_tasks),
            fields(
                gpu_id = %gpu,
                block = %request.block,
                $($field = %request.$field),*
            )
        )]
        pub async fn [<$action:snake _ $api:snake _handler>](
            gpu: web::Path<usize>,
            request: web::Json<[<$action:camel $api:camel>]>,
            device_tasks: web::Data<GpuManagementTasks>,
        ) -> Result<HttpResponse, ZeusdError> {
            let gpu = gpu.into_inner();
            let request = request.into_inner();

            tracing::info!(
                "Received reqeust to GPU {} ({:?})",
                gpu,
                request,
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
        }
    };
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SetPersistentMode {
    pub enabled: bool,
    pub block: bool,
}

impl_handler_for_gpu_command!(set, persistent_mode, "/{gpu_id}/set_persistent_mode", enabled);

#[derive(Serialize, Deserialize, Debug)]
pub struct SetPowerLimit {
    power_limit_mw: u32,
    block: bool,
}

impl_handler_for_gpu_command!(set, power_limit, "/{gpu_id}/set_power_limit", power_limit_mw);

#[derive(Serialize, Deserialize, Debug)]
pub struct SetGpuLockedClocks {
    min_clock_mhz: u32,
    max_clock_mhz: u32,
    block: bool,
}

impl_handler_for_gpu_command!(
    set,
    gpu_locked_clocks,
    "/{gpu_id}/set_gpu_locked_clocks",
    min_clock_mhz,
    max_clock_mhz
);

#[derive(Serialize, Deserialize, Debug)]
pub struct SetMemLockedClocks {
    min_clock_mhz: u32,
    max_clock_mhz: u32,
    block: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResetGpuLockedClocks {
    block: bool,
}

impl_handler_for_gpu_command!(
    reset,
    gpu_locked_clocks,
    "/{gpu_id}/reset_gpu_locked_clocks",
);


impl_handler_for_gpu_command!(
    set,
    mem_locked_clocks,
    "/{gpu_id}/set_mem_locked_clocks",
    min_clock_mhz,
    max_clock_mhz
);
#[derive(Serialize, Deserialize, Debug)]
pub struct ResetMemLockedClocks {
    block: bool,
}

impl_handler_for_gpu_command!(
    reset,
    mem_locked_clocks,
    "/{gpu_id}/reset_mem_locked_clocks",
);

pub fn gpu_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(set_persistent_mode_handler)
        .service(set_power_limit_handler)
        .service(set_gpu_locked_clocks_handler)
        .service(reset_gpu_locked_clocks_handler)
        .service(set_mem_locked_clocks_handler)
        .service(reset_mem_locked_clocks_handler);
}
