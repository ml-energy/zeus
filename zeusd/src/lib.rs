//! Zeus daemon library.

#[cfg(all(feature = "amdsmi", not(target_os = "linux")))]
compile_error!("the amdsmi feature is only supported on Linux");

pub mod auth;
pub mod config;
pub mod devices;
pub mod error;
pub mod power_streaming;
pub mod routes;
pub mod startup;
