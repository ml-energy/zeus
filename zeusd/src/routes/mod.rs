//! Routes and handlers for interacting with devices

pub mod cpu;
pub mod gpu;

pub use cpu::cpu_routes;
pub use gpu::gpu_routes;
