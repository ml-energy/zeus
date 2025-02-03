//! Routes for interacting with GPUs

use actix_web::{web, HttpResponse};
use paste::paste;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::devices::gpu::{GpuCommand, GpuManagementTasks};
use crate::error::ZeusdError;

/// Macro to generate a handler for a GPU command.
///
/// This macro takes
/// - the API name (set_power_limit, set_persistence_mode, etc.),
/// - the method and path for the request handler,
/// - and a list of `field name: type` pairs of the corresponding `GpuCommand` variant.
///
/// Gien this, the macro generates
/// - a request payload struct named API name (e.g., SetPowerLimit) and all the
///     fields specified plus `block: bool` to indicate whether the request should block,
/// - an implementation of `From` for the payload struct to convert it to the
/// - a handler function that takes the request payload, converts it to a `GpuCommand` variant,
///     and sends it to the `GpuManagementTasks` actor.
///
///  Assumptions:
///  - The `GpuCommand` variant name is the same as the API name, but the former is camel case
///      and the latter is snake case (e.g., SetPowerLimit vs. set_power_limit).
macro_rules! impl_handler_for_gpu_command {
    ($api:ident, $path:expr, $($field:ident: $ftype:ty,)*) => {
        paste! {
        // Request payload structure.
        #[derive(Serialize, Deserialize, Debug)]
        pub struct [<$api:camel>] {
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
        #[actix_web::$path]
        #[tracing::instrument(
            skip(request, device_tasks),
            fields(
                gpu_id = %gpu_id,
                block = %request.block,
                $($field = %request.$field),*
            )
        )]
        async fn [<$api:snake _handler>](
            gpu_id: web::Path<usize>,
            request: web::Json<[<$api:camel>]>,
            device_tasks: web::Data<GpuManagementTasks>,
        ) -> Result<HttpResponse, ZeusdError> {
            let now = Instant::now();

            tracing::info!("Received request");

            let gpu_id = gpu_id.into_inner();
            let request = request.into_inner();

            if request.block {
                device_tasks.send_command_blocking(gpu_id, request.into(), now).await?;
            } else {
                device_tasks.send_command_nonblocking(gpu_id, request.into(), now)?;
            }

            Ok(HttpResponse::Ok().finish())
        }
        }
    };
}

impl_handler_for_gpu_command!(
    set_persistence_mode,
    post("/{gpu_id}/set_persistence_mode"),
    enabled: bool,
);

impl_handler_for_gpu_command!(
    set_power_limit,
    post("/{gpu_id}/set_power_limit"),
    power_limit_mw: u32,
);

impl_handler_for_gpu_command!(
    set_gpu_locked_clocks,
    post("/{gpu_id}/set_gpu_locked_clocks"),
    min_clock_mhz: u32,
    max_clock_mhz: u32,
);

impl_handler_for_gpu_command!(
    reset_gpu_locked_clocks,
    post("/{gpu_id}/reset_gpu_locked_clocks"),
);

impl_handler_for_gpu_command!(
    set_mem_locked_clocks,
    post("/{gpu_id}/set_mem_locked_clocks"),
    min_clock_mhz: u32,
    max_clock_mhz: u32,
);

impl_handler_for_gpu_command!(
    reset_mem_locked_clocks,
    post("/{gpu_id}/reset_mem_locked_clocks"),
);

/// Register GPU routes with the Actix web server.
pub fn gpu_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(set_persistence_mode_handler)
        .service(set_power_limit_handler)
        .service(set_gpu_locked_clocks_handler)
        .service(reset_gpu_locked_clocks_handler)
        .service(set_mem_locked_clocks_handler)
        .service(reset_mem_locked_clocks_handler);
}
