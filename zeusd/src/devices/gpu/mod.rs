//! GPU management

// NVIDIA GPUs
mod nvml;
pub use nvml::*;

// Real NVML interface.
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::NvmlGpu;

// Fake NVML interface for dev and testing on macOS.
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub use macos::NvmlGpu;
