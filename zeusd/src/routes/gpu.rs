//! Routes for interacting with GPUs

use actix_web::{web, HttpResponse};
use paste::paste;
use serde::{Deserialize, Serialize};

use crate::devices::gpu::{GpuCommand, GpuManagementTasks};
use crate::error::ZeusdError;

/// Macro to generate a handler for a GPU command.
///
/// This macro takes
/// - the action (set, reset, etc.),
/// - the API name (power_limit, persistent_mode, etc.),
/// - the method and path for the request handler,
/// - and a list of `field name <type>` pairs of the corresponding `GpuCommand` variant.
///
/// Gien this, the macro generates
/// - a request payload struct named action + API name (e.g., SetPowerLimit) and all the
///  fields specified plus `block: bool` to indicate whether the request should block,
/// - an implementation of `From` for the payload struct to convert it to the
/// - a handler function that takes the request payload, converts it to a `GpuCommand` variant,
///  and sends it to the `GpuManagementTasks` actor.
///
///  Assumptions:
///  - The `GpuCommand` variant name is a concatenation of the action and API name
///   (e.g., set and power_limit -> SetPowerLimit).
macro_rules! impl_handler_for_gpu_command {
    ($action:ident, $api:ident, $path:expr, $($field:ident <$ftype:ty>,)*) => {
        paste! {
        // Request payload structure.
        #[derive(Serialize, Deserialize, Debug)]
        pub struct [<$action:camel $api:camel>] {
            $(pub $field: $ftype,)*
            pub block: bool,
        }

        // Implement conversion to the GpuCommand variant.
        impl From<[<$action:camel $api:camel>]> for GpuCommand {
            // Prefixing with underscore to avoid lint errors when $field is empty.
            fn from(_request: [<$action:camel $api:camel>]) -> Self {
                GpuCommand::[<$action:camel $api:camel>] {
                    $($field: _request.$field),*
                }
            }
        }

        // Generate the request handler.
        #[actix_web::$path]
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

impl_handler_for_gpu_command!(
    set,
    persistent_mode,
    post("/{gpu_id}/set_persistent_mode"),
    enabled<bool>,
);

impl_handler_for_gpu_command!(
    set,
    power_limit,
    post("/{gpu_id}/set_power_limit"),
    power_limit_mw<u32>,
);

impl_handler_for_gpu_command!(
    set,
    gpu_locked_clocks,
    post("/{gpu_id}/set_gpu_locked_clocks"),
    min_clock_mhz<u32>,
    max_clock_mhz<u32>,
);

impl_handler_for_gpu_command!(
    reset,
    gpu_locked_clocks,
    post("/{gpu_id}/reset_gpu_locked_clocks"),
);

impl_handler_for_gpu_command!(
    set,
    mem_locked_clocks,
    post("/{gpu_id}/set_mem_locked_clocks"),
    min_clock_mhz<u32>,
    max_clock_mhz<u32>,
);

impl_handler_for_gpu_command!(
    reset,
    mem_locked_clocks,
    post("/{gpu_id}/reset_mem_locked_clocks"),
);

pub fn gpu_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(set_persistent_mode_handler)
        .service(set_power_limit_handler)
        .service(set_gpu_locked_clocks_handler)
        .service(reset_gpu_locked_clocks_handler)
        .service(set_mem_locked_clocks_handler)
        .service(reset_mem_locked_clocks_handler);
}
