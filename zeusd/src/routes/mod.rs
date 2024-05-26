pub mod gpu;

use actix_web::web;
use gpu::{set_persistent_mode_handler, set_power_limit_handler, set_gpu_locked_clocks_handler, set_mem_locked_clocks_handler};

pub fn gpu_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(set_persistent_mode_handler)
        .service(set_power_limit_handler)
        .service(set_gpu_locked_clocks_handler)
        .service(set_mem_locked_clocks_handler);
}
