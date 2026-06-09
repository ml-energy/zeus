//! Error handling.
//!
//! This module defines the `ZeusdError` enum, which is used to represent errors
//! that can occur when handling requests to the Zeus daemon.
//!
//! Note that errors that occur during the initialization of the daemon are
//! handled with `anyhow` and eventually end up terminating the process.

use std::collections::HashMap;

use actix_web::http::StatusCode;
use actix_web::{HttpResponse, ResponseError};
use nvml_wrapper::error::NvmlError;
use tokio::sync::mpsc::error::SendError;

use crate::devices::cpu::CpuCommandRequest;
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
    #[error("CPU command send error: {0}")]
    CpuCommandSendError(#[from] SendError<CpuCommandRequest>),
    #[error("Management task for GPU {0} unexpectedly terminated while handling the request.")]
    GpuManagementTaskTerminatedError(usize),
    #[error("Management task for CPU {0} unexpectedly terminated while handling the request.")]
    CpuManagementTaskTerminatedError(usize),
    #[error("CPU {0} did not return the energy data required for power measurement.")]
    CpuPowerMeasurementError(usize),
    #[error("Initialization for CPU {0} unexpectedly errored.")]
    CpuInitializationError(usize),
    #[error("IOError: {0}")]
    IOError(#[from] std::io::Error),
    #[error("Authentication required.")]
    Unauthorized,
    #[error("Insufficient permissions: {0}")]
    Forbidden(String),
    #[error("Persistence mode cannot be disabled on this platform.")]
    PersistenceModeCannotBeDisabled,
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
                NvmlError::NotSupported => StatusCode::BAD_REQUEST,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            },
            ZeusdError::GpuCommandSendError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::CpuCommandSendError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::GpuManagementTaskTerminatedError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::CpuManagementTaskTerminatedError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::CpuPowerMeasurementError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::CpuInitializationError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::IOError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::Unauthorized => StatusCode::UNAUTHORIZED,
            ZeusdError::Forbidden(_) => StatusCode::FORBIDDEN,
            ZeusdError::PersistenceModeCannotBeDisabled => StatusCode::BAD_REQUEST,
        }
    }
}

/// Aggregate per-device errors into one response, status = max(status_code).
pub fn aggregate_error_response(errors: HashMap<usize, ZeusdError>) -> HttpResponse {
    let worst_status = errors
        .values()
        .map(|e| e.status_code())
        .max()
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let payload: HashMap<String, String> = errors
        .into_iter()
        .map(|(id, e)| (id.to_string(), e.to_string()))
        .collect();
    HttpResponse::build(worst_status).json(serde_json::json!({ "errors": payload }))
}
