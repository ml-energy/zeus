//! Routes and handlers for interacting with devices

pub mod cpu;
pub mod gpu;
pub mod server;

pub use cpu::cpu_routes;
pub use gpu::{gpu_control_routes, gpu_read_routes};
pub use server::{server_routes, DiscoveryInfo};
