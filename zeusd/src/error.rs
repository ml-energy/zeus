//! Error handling.
//!
//! This module defines the `ZeusdError` enum, which is used to represent errors
//! that can occur when handling requests to the Zeus daemon.
//!
//! Note that errors that occur during the initialization of the daemon are
//! handled with `anyhow` and eventually end up terminating the process.

use actix_web::http::StatusCode;
use actix_web::ResponseError;
use nvml_wrapper::error::NvmlError;
use tokio::sync::mpsc::error::SendError;

use crate::devices::gpu::GpuCommandRequest;

#[derive(thiserror::Error, Debug)]
pub enum ZeusdError {
    #[error("GPU index {0} does not exist.")]
    GpuNotFoundError(usize),
    #[error("CPU index {0} does not exist.")]
    CpuNotFoundError(usize),
    #[error("NVML error: {0}")]
    NvmlError(#[from] NvmlError),
    #[error("GPU command send error: {0}")]
    GpuCommandSendError(#[from] SendError<GpuCommandRequest>),
    #[error("Management task for GPU {0} unexpectedly terminated while handling the request.")]
    GpuManagementTaskTerminatedError(usize),
    #[error("Management task for CPU {0} unexpectedly terminated while handling the request.")]
    CpuManagementTaskTerminatedError(usize),
    #[error("Initialization for CPU {0} unexpectedly errored.")]
    CpuInitializationError(usize),
    #[error("IOError: {0}")]
    IOError(#[from] std::io::Error),
}

/// This allows us to return a custom HTTP status code for each error variant.
impl ResponseError for ZeusdError {
    fn status_code(&self) -> StatusCode {
        match self {
            ZeusdError::GpuNotFoundError(_) => StatusCode::BAD_REQUEST,
            ZeusdError::CpuNotFoundError(_) => StatusCode::BAD_REQUEST,
            ZeusdError::NvmlError(e) => match e {
                NvmlError::NoPermission => StatusCode::FORBIDDEN,
                NvmlError::InvalidArg => StatusCode::BAD_REQUEST,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            },
            ZeusdError::GpuCommandSendError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::GpuManagementTaskTerminatedError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::CpuManagementTaskTerminatedError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::CpuInitializationError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::IOError(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}
