//! Routes and handlers for interacting with devices

pub mod cpu;
pub mod gpu;

pub use cpu::cpu_routes;
pub use gpu::gpu_routes;

/// Parse a comma-separated list of device indices from a query parameter.
pub fn parse_device_ids(raw: &Option<String>) -> Option<Vec<usize>> {
    raw.as_ref().map(|s| {
        s.split(',')
            .filter_map(|part| part.trim().parse().ok())
            .collect()
    })
}
