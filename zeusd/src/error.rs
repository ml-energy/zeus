//! Error handling.
//!
//! This module defines the `ZeusdError` enum, which is used to represent errors
//! that can occur when handling requests to the Zeus daemon.
//!
//! Note that errors that occur during the initialization of the daemon are
//! handled with `anyhow` and eventually end up terminating the process.

use std::collections::HashMap;

#[cfg(feature = "amdsmi")]
use crate::devices::gpu::amdsmi::ffi::{
    AMDSMI_STATUS_INVAL, AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM,
};
use actix_web::http::StatusCode;
use actix_web::{HttpResponse, ResponseError};
#[cfg(feature = "nvml")]
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
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[cfg(feature = "nvml")]
    #[error("NVML error: {0}")]
    NvmlError(#[from] NvmlError),
    #[cfg(feature = "amdsmi")]
    #[error("AMDSMI error {status}: {msg}")]
    AmdSmiError { status: u32, msg: String },
    #[cfg(feature = "amdsmi")]
    #[error("Failed to load AMD SMI: {0}")]
    AmdSmiLoadError(String),
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
    #[error("Management task for CPU {0} returned a response of the wrong type.")]
    CpuUnexpectedResponseError(usize),
    #[error("Initialization for CPU {0} unexpectedly errored.")]
    CpuInitializationError(usize),
    #[error("Failed to {action}: {source}")]
    CpuControlError {
        action: String,
        source: std::io::Error,
    },
    #[error("No power limit baseline was recorded for CPU {0}.")]
    CpuBaselineMissingError(usize),
    #[error(
        "Cannot {action} on CPU {cpu}: {source} {}",
        crate::devices::cpu::msr::MSR_AVAILABILITY
    )]
    CpuMsrError {
        cpu: usize,
        action: &'static str,
        source: crate::devices::cpu::msr::MsrError,
    },
    #[error("{}", .0.iter().map(ToString::to_string).collect::<Vec<_>>().join("; "))]
    Multiple(Vec<ZeusdError>),
    #[error("IOError: {0}")]
    IOError(#[from] std::io::Error),
    #[error("Authentication required.")]
    Unauthorized,
    #[error("Insufficient permissions: {0}")]
    Forbidden(String),
    #[error("Persistence mode cannot be disabled on this platform.")]
    PersistenceModeCannotBeDisabled,
    #[error("Override command failed: {0}")]
    CommandOverrideError(String),
}

/// This allows us to return a custom HTTP status code for each error variant.
impl ResponseError for ZeusdError {
    fn status_code(&self) -> StatusCode {
        match self {
            ZeusdError::GpuNotFoundError(_) => StatusCode::BAD_REQUEST,
            ZeusdError::CpuNotFoundError(_) => StatusCode::BAD_REQUEST,
            ZeusdError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            #[cfg(feature = "nvml")]
            ZeusdError::NvmlError(e) => match e {
                NvmlError::NoPermission => StatusCode::FORBIDDEN,
                NvmlError::InvalidArg => StatusCode::BAD_REQUEST,
                NvmlError::NotSupported => StatusCode::BAD_REQUEST,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            },
            #[cfg(feature = "amdsmi")]
            ZeusdError::AmdSmiError { status, .. } => match *status {
                AMDSMI_STATUS_NO_PERM => StatusCode::FORBIDDEN,
                AMDSMI_STATUS_INVAL | AMDSMI_STATUS_NOT_SUPPORTED => StatusCode::BAD_REQUEST,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            },
            #[cfg(feature = "amdsmi")]
            ZeusdError::AmdSmiLoadError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::GpuCommandSendError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::CpuCommandSendError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::GpuManagementTaskTerminatedError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::CpuManagementTaskTerminatedError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::CpuPowerMeasurementError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::CpuUnexpectedResponseError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::CpuInitializationError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::CpuControlError { source, .. } => cpu_control_status(source),
            ZeusdError::CpuBaselineMissingError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::CpuMsrError { source, .. } => {
                use crate::devices::cpu::msr::MsrError;
                match source {
                    MsrError::InvalidWindow(_) => StatusCode::BAD_REQUEST,
                    MsrError::PermissionDenied(_)
                    | MsrError::WriteDenied { .. }
                    | MsrError::Locked => StatusCode::FORBIDDEN,
                    MsrError::DriverMissing(_) | MsrError::UnsupportedLayout => {
                        StatusCode::SERVICE_UNAVAILABLE
                    }
                    _ => StatusCode::INTERNAL_SERVER_ERROR,
                }
            }
            ZeusdError::Multiple(errors) => errors
                .iter()
                .map(ResponseError::status_code)
                .max()
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            ZeusdError::IOError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::Unauthorized => StatusCode::UNAUTHORIZED,
            ZeusdError::Forbidden(_) => StatusCode::FORBIDDEN,
            ZeusdError::PersistenceModeCannotBeDisabled => StatusCode::BAD_REQUEST,
            ZeusdError::CommandOverrideError(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl ZeusdError {
    /// Report an unavailable or failed MSR operation in the response and daemon log.
    pub fn cpu_msr(
        cpu: usize,
        action: &'static str,
        source: crate::devices::cpu::msr::MsrError,
    ) -> Self {
        let error = Self::CpuMsrError {
            cpu,
            action,
            source,
        };
        tracing::warn!("{error}");
        error
    }
    /// A failed CPU power limit or time window write.
    pub fn cpu_control(action: String, source: std::io::Error) -> Self {
        ZeusdError::CpuControlError { action, source }
    }

    /// Succeed if `errors` is empty, and otherwise fail with all of them.
    pub fn from_errors(mut errors: Vec<ZeusdError>) -> Result<(), ZeusdError> {
        match errors.len() {
            0 => Ok(()),
            1 => Err(errors.remove(0)),
            _ => Err(ZeusdError::Multiple(errors)),
        }
    }
}

/// Map the errno of a failed CPU control write to an HTTP status.
///
/// The kernel answers `EACCES` for RAPL limits the BIOS locked and `EPERM` for
/// HSMP writes through a read-only file. HSMP answers `EINVAL` for arguments
/// the firmware rejects and `ENOMSG` for messages the firmware does not know.
fn cpu_control_status(source: &std::io::Error) -> StatusCode {
    #[cfg(unix)]
    {
        use nix::errno::Errno;
        match source.raw_os_error().map(Errno::from_raw) {
            Some(Errno::EACCES | Errno::EPERM) => StatusCode::FORBIDDEN,
            Some(Errno::EINVAL | Errno::ENOMSG) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = source;
        StatusCode::INTERNAL_SERVER_ERROR
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn control_error(errno: i32) -> ZeusdError {
        ZeusdError::cpu_control(
            "set a power limit".to_string(),
            std::io::Error::from_raw_os_error(errno),
        )
    }

    #[cfg(unix)]
    #[test]
    fn cpu_control_errno_maps_to_status() {
        use nix::errno::Errno;
        for (errno, status) in [
            (Errno::EACCES, StatusCode::FORBIDDEN),
            (Errno::EPERM, StatusCode::FORBIDDEN),
            (Errno::EINVAL, StatusCode::BAD_REQUEST),
            (Errno::ENOMSG, StatusCode::BAD_REQUEST),
            (Errno::EIO, StatusCode::INTERNAL_SERVER_ERROR),
            (Errno::ETIMEDOUT, StatusCode::INTERNAL_SERVER_ERROR),
        ] {
            assert_eq!(control_error(errno as i32).status_code(), status, "{errno}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn multiple_errors_report_all_and_the_worst_status() {
        use nix::errno::Errno;
        assert!(ZeusdError::from_errors(vec![]).is_ok());

        let error = ZeusdError::from_errors(vec![
            control_error(Errno::EINVAL as i32),
            control_error(Errno::EACCES as i32),
        ])
        .unwrap_err();
        assert_eq!(error.status_code(), StatusCode::FORBIDDEN);
        assert_eq!(error.to_string().matches("set a power limit").count(), 2);
    }

    #[test]
    fn msr_errors_explain_available_operations_and_required_access() {
        use crate::devices::cpu::msr::MsrError;
        for (source, status, remedy) in [
            (
                MsrError::DriverMissing("/dev/cpu/0/msr".into()),
                StatusCode::SERVICE_UNAVAILABLE,
                "sudo modprobe msr",
            ),
            (
                MsrError::PermissionDenied("/dev/cpu/0/msr".into()),
                StatusCode::FORBIDDEN,
                "CAP_SYS_RAWIO",
            ),
            (
                MsrError::WriteDenied {
                    path: "/dev/cpu/0/msr".into(),
                    source: std::io::ErrorKind::PermissionDenied.into(),
                },
                StatusCode::FORBIDDEN,
                "msr.allow_writes",
            ),
            (MsrError::Locked, StatusCode::FORBIDDEN, "BIOS"),
            (
                MsrError::UnsupportedLayout,
                StatusCode::SERVICE_UNAVAILABLE,
                "unsupported",
            ),
            (
                MsrError::InvalidWindow("too large".into()),
                StatusCode::BAD_REQUEST,
                "too large",
            ),
        ] {
            let error = ZeusdError::cpu_msr(0, "set the time window", source);
            assert_eq!(error.status_code(), status);
            let message = error.to_string();
            assert!(message.contains(remedy), "{message}");
            assert!(
                message.contains("power-limit changes remain available"),
                "{message}"
            );
            assert!(message.contains("require MSR write access"), "{message}");
            assert!(
                message.contains("AMD HSMP control does not require MSR access"),
                "{message}"
            );
        }
    }
}
