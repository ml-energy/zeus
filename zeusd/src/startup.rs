use std::os::unix::net::UnixListener;
use actix_web::{dev::Server, web, App, HttpServer};

use crate::device::gpu::GpuHandlers;
use crate::routes::gpu::{set_frequency, set_power_limit};
use crate::routes::info::info;

pub fn run(listener: UnixListener, gpu_handlers: GpuHandlers) -> Server {
    let server = HttpServer::new(move || {
        App::new()
            .service(info)
            .service(set_power_limit)
            .service(set_frequency)
            .app_data(web::Data::new(gpu_handlers.clone()))
    })
    .listen_uds(listener)
    .expect("Failed to bind to socket.")
    .run();

    server
}
