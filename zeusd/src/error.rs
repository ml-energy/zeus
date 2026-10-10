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

use crate::devices::cpu::hsmp::HsmpError;
use crate::devices::cpu::CpuCommandRequest;
use crate::devices::gpu::GpuCommandRequest;

/// Documentation of the prerequisites and permissions each Zeusd feature needs.
pub const PERMISSIONS_DOC_URL: &str =
    "https://ml.energy/zeus/zeusd/deployment/#feature-requirements-and-permissions";

/// Documentation of the original CPU power limits that reset restores.
pub const ORIGINAL_POWER_LIMITS_DOC_URL: &str =
    "https://ml.energy/zeus/zeusd/deployment/#original-cpu-power-limits";

/// How to record the original setting of a constraint that the recorded original settings lack.
pub const RECORD_ORIGINAL_HINT: &str = "To record it after restoring access, reset the recorded \
    settings and stop Zeusd. Verify that the current limits and time windows are the values \
    reset should restore. Restart with a new --original-cpu-power-limit-path, or with \
    --no-persistent-original-cpu-power-limit to record originals at each daemon start.";

/// How to make the RAPL powercap interface available to Zeusd.
pub const RAPL_AVAILABILITY: &str = "Ensure the host's RAPL powercap interface is available, \
    typically provided by the intel_rapl_msr kernel module (`sudo modprobe intel_rapl_msr`). In \
    a container, bind-mount the host's /sys/class/powercap at /zeus_sys/class/powercap and \
    /sys/devices/virtual/powercap at /zeus_sys/devices/virtual/powercap.";

#[derive(thiserror::Error, Debug)]
pub enum ZeusdError {
    #[error("GPU index {0} does not exist.")]
    GpuNotFoundError(usize),
    #[error("CPU index {0} does not exist.")]
    CpuNotFoundError(usize),
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[cfg(feature = "nvml")]
    #[error("NVML error: {0}{}", nvml_hint(.0))]
    NvmlError(#[from] NvmlError),
    #[cfg(feature = "amdsmi")]
    #[error("AMDSMI error {status}: {msg}{}", amdsmi_hint(*.status))]
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
    #[error(
        "Cannot initialize RAPL for CPU {cpu}: {reason}. {RAPL_AVAILABILITY} \
         See {PERMISSIONS_DOC_URL}"
    )]
    CpuInitializationError { cpu: usize, reason: String },
    #[error(
        "Cannot read the RAPL energy counter {path} of CPU {cpu}: {source}.{}",
        energy_read_hint(.source)
    )]
    CpuEnergyReadError {
        cpu: usize,
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error(
        "Cannot read the RAPL power limit file {path}: {source}.{} See {PERMISSIONS_DOC_URL}",
        limit_read_hint(.source)
    )]
    CpuLimitReadError {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("Failed to {action}: {source}.{}", cpu_control_hint(.source))]
    CpuControlError {
        action: String,
        source: std::io::Error,
    },
    #[error("Cannot {action} on CPU {cpu}: {source} See {PERMISSIONS_DOC_URL}")]
    CpuHsmpError {
        cpu: usize,
        action: String,
        /// Whether the failed operation changes a setting.
        write: bool,
        source: HsmpError,
    },
    #[error(
        "No original power limit settings were recorded for CPU {0}, so they cannot be restored. \
         Zeusd records them at startup when the cpu-control API group is enabled. \
         See {ORIGINAL_POWER_LIMITS_DOC_URL}"
    )]
    CpuOriginalPowerLimitsMissingError(usize),
    #[error(
        "No original setting of constraint '{constraint}' is recorded for CPU {cpu} because the \
         constraint was unavailable when Zeusd recorded the original settings, and Zeusd does not \
         change a constraint that reset cannot restore. {RECORD_ORIGINAL_HINT} \
         See {ORIGINAL_POWER_LIMITS_DOC_URL}"
    )]
    CpuConstraintOriginalMissingError { cpu: usize, constraint: String },
    #[error(
        "Cannot {action} on CPU {cpu}: {source} {} See {PERMISSIONS_DOC_URL}",
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
            ZeusdError::CpuInitializationError { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ZeusdError::CpuEnergyReadError { source, .. }
            | ZeusdError::CpuLimitReadError { source, .. } => match source.kind() {
                std::io::ErrorKind::PermissionDenied => StatusCode::FORBIDDEN,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            },
            ZeusdError::CpuControlError { source, .. } => cpu_control_status(source),
            ZeusdError::CpuHsmpError { write, source, .. } => match source {
                HsmpError::DeviceMissing(_) => StatusCode::SERVICE_UNAVAILABLE,
                HsmpError::PermissionDenied { .. } => StatusCode::FORBIDDEN,
                HsmpError::Open { .. } => StatusCode::INTERNAL_SERVER_ERROR,
                HsmpError::Request(source)
                    if source.kind() == std::io::ErrorKind::PermissionDenied =>
                {
                    StatusCode::FORBIDDEN
                }
                HsmpError::Request(source) if *write => cpu_control_status(source),
                HsmpError::Request(_) => StatusCode::INTERNAL_SERVER_ERROR,
            },
            ZeusdError::CpuOriginalPowerLimitsMissingError(_)
            | ZeusdError::CpuConstraintOriginalMissingError { .. } => {
                StatusCode::SERVICE_UNAVAILABLE
            }
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
    /// A failed RAPL power limit write through the powercap sysfs interface.
    pub fn cpu_control(action: String, source: std::io::Error) -> Self {
        ZeusdError::CpuControlError { action, source }
    }

    /// A failed HSMP read (`write` false) or setting change (`write` true).
    pub fn cpu_hsmp(cpu: usize, action: impl Into<String>, write: bool, source: HsmpError) -> Self {
        ZeusdError::CpuHsmpError {
            cpu,
            action: action.into(),
            write,
            source,
        }
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
/// The kernel answers `EACCES` both for RAPL limits the BIOS locked and for
/// missing file permissions, and `EROFS` when sysfs is mounted read-only.
/// HSMP answers `EINVAL` for arguments the firmware rejects and `ENOMSG` for
/// messages the firmware does not know.
fn cpu_control_status(source: &std::io::Error) -> StatusCode {
    #[cfg(unix)]
    {
        use nix::errno::Errno;
        match source.raw_os_error().map(Errno::from_raw) {
            Some(Errno::EACCES | Errno::EPERM | Errno::EROFS) => StatusCode::FORBIDDEN,
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

/// Explain the possible causes of a failed RAPL powercap write.
fn cpu_control_hint(source: &std::io::Error) -> String {
    #[cfg(unix)]
    {
        use nix::errno::Errno;
        match source.raw_os_error().map(Errno::from_raw) {
            Some(Errno::EACCES | Errno::EPERM) => format!(
                " The kernel denied the write. This happens when Zeusd lacks write permission on \
                 the root-owned powercap sysfs file or when the BIOS locked the constraint; the \
                 kernel does not report which. See {PERMISSIONS_DOC_URL}"
            ),
            Some(Errno::EROFS) => format!(
                " sysfs is mounted read-only for Zeusd. Under systemd, ProtectKernelTunables=true \
                 does this; in a container, mount the powercap directory read-write. \
                 See {PERMISSIONS_DOC_URL}"
            ),
            _ => String::new(),
        }
    }
    #[cfg(not(unix))]
    {
        let _ = source;
        String::new()
    }
}

/// Explain the prerequisites of reading a RAPL energy counter.
fn energy_read_hint(source: &std::io::Error) -> String {
    match source.kind() {
        std::io::ErrorKind::PermissionDenied => format!(
            " The kernel makes RAPL energy counters readable only by their owner, root \
             (CVE-2020-8694). Run Zeusd as root, or give a non-root Zeusd read access to the \
             file, for example through a permission change by the administrator or \
             CAP_DAC_READ_SEARCH. See {PERMISSIONS_DOC_URL}"
        ),
        std::io::ErrorKind::NotFound => format!(
            " The RAPL zone no longer exists, for example because the RAPL driver was \
             unloaded. See {PERMISSIONS_DOC_URL}"
        ),
        _ => String::new(),
    }
}

/// Explain a failed read of a RAPL powercap zone or constraint file.
fn limit_read_hint(source: &std::io::Error) -> &'static str {
    match source.kind() {
        std::io::ErrorKind::PermissionDenied => {
            " Zeusd lacks read permission on this file. Power limit queries and control read the \
             zone's `enabled` and `constraint_*` files, so give the user Zeusd runs as read \
             access to them."
        }
        std::io::ErrorKind::NotFound => {
            " The RAPL zone or this file no longer exists, for example because the RAPL driver \
             was unloaded."
        }
        _ => "",
    }
}

#[cfg(feature = "nvml")]
fn nvml_hint(error: &NvmlError) -> String {
    match error {
        NvmlError::NoPermission if cfg!(windows) => format!(
            ". NVML reported that Zeusd lacks permission for this operation. NVIDIA GPU control \
             requires administrator rights, so run Zeusd from an elevated shell. \
             See {PERMISSIONS_DOC_URL}"
        ),
        NvmlError::NoPermission => format!(
            ". NVML reported that Zeusd lacks permission for this operation. Check NVIDIA device \
             permissions and container device access. NVIDIA GPU control \
             generally requires root, and with a restricted capability set, such as under \
             systemd or in a container, also CAP_SYS_ADMIN; NVML does not report which is \
             missing. See {PERMISSIONS_DOC_URL}"
        ),
        _ => String::new(),
    }
}

#[cfg(feature = "amdsmi")]
fn amdsmi_hint(status: u32) -> String {
    match status {
        AMDSMI_STATUS_NO_PERM => format!(
            ". AMD SMI reported that Zeusd lacks permission for this operation. Check GPU device \
             access and read permission on GPU sysfs files. AMD GPU control \
             writes sysfs files, which requires root and a writable /sys (under systemd, \
             ProtectKernelTunables=false); AMD SMI does not report which is missing. \
             See {PERMISSIONS_DOC_URL}"
        ),
        _ => String::new(),
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
            (Errno::EROFS, StatusCode::FORBIDDEN),
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
    fn hsmp_errors_map_to_status() {
        use nix::errno::Errno;
        use std::path::PathBuf;
        let denied = |write| HsmpError::PermissionDenied {
            path: PathBuf::from("/dev/hsmp"),
            write,
            source: std::io::ErrorKind::PermissionDenied.into(),
        };
        let request =
            |errno: Errno| HsmpError::Request(std::io::Error::from_raw_os_error(errno as i32));
        for (source, write, status) in [
            (
                HsmpError::DeviceMissing(PathBuf::from("/dev/hsmp")),
                false,
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (denied(false), false, StatusCode::FORBIDDEN),
            (denied(true), true, StatusCode::FORBIDDEN),
            (request(Errno::EINVAL), true, StatusCode::BAD_REQUEST),
            (
                request(Errno::EINVAL),
                false,
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
            (request(Errno::EACCES), false, StatusCode::FORBIDDEN),
            (request(Errno::EACCES), true, StatusCode::FORBIDDEN),
            (request(Errno::EPERM), false, StatusCode::FORBIDDEN),
            (request(Errno::EPERM), true, StatusCode::FORBIDDEN),
        ] {
            let error = ZeusdError::cpu_hsmp(0, "use HSMP", write, source);
            assert_eq!(error.status_code(), status, "{error}");
        }
    }

    #[test]
    fn rapl_read_permission_errors_are_forbidden() {
        let path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj";
        let energy = |kind: std::io::ErrorKind| ZeusdError::CpuEnergyReadError {
            cpu: 0,
            path: path.into(),
            source: kind.into(),
        };
        let limit = |kind: std::io::ErrorKind| ZeusdError::CpuLimitReadError {
            path: path.into(),
            source: kind.into(),
        };
        use std::io::ErrorKind::{InvalidData, NotFound, PermissionDenied};
        for (error, status) in [
            (energy(PermissionDenied), StatusCode::FORBIDDEN),
            (energy(NotFound), StatusCode::INTERNAL_SERVER_ERROR),
            (limit(PermissionDenied), StatusCode::FORBIDDEN),
            (limit(InvalidData), StatusCode::INTERNAL_SERVER_ERROR),
        ] {
            assert_eq!(error.status_code(), status, "{error}");
        }
    }

    #[test]
    fn missing_original_is_unavailable() {
        let error = ZeusdError::CpuOriginalPowerLimitsMissingError(0);
        assert_eq!(error.status_code(), StatusCode::SERVICE_UNAVAILABLE);

        let error = ZeusdError::CpuConstraintOriginalMissingError {
            cpu: 0,
            constraint: "socket".to_string(),
        };
        assert_eq!(error.status_code(), StatusCode::SERVICE_UNAVAILABLE);
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

    #[cfg(feature = "nvml")]
    #[test]
    fn nvml_permission_error_uses_platform_requirements() {
        let error = ZeusdError::from(NvmlError::NoPermission);
        assert_eq!(error.status_code(), StatusCode::FORBIDDEN);
        let requirement = if cfg!(windows) {
            "elevated shell"
        } else {
            "CAP_SYS_ADMIN"
        };
        assert!(error.to_string().contains(requirement));
        assert!(!ZeusdError::from(NvmlError::NotSupported)
            .to_string()
            .contains(PERMISSIONS_DOC_URL));
    }

    #[cfg(feature = "amdsmi")]
    #[test]
    fn amdsmi_permission_error_is_forbidden() {
        let error = ZeusdError::AmdSmiError {
            status: AMDSMI_STATUS_NO_PERM,
            msg: "set power cap".to_string(),
        };
        assert_eq!(error.status_code(), StatusCode::FORBIDDEN);
    }

    #[test]
    fn msr_errors_map_to_status() {
        use crate::devices::cpu::msr::MsrError;
        for (source, status) in [
            (
                MsrError::DriverMissing("/dev/cpu/0/msr".into()),
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                MsrError::PermissionDenied("/dev/cpu/0/msr".into()),
                StatusCode::FORBIDDEN,
            ),
            (
                MsrError::WriteDenied {
                    path: "/dev/cpu/0/msr".into(),
                    source: std::io::ErrorKind::PermissionDenied.into(),
                },
                StatusCode::FORBIDDEN,
            ),
            (MsrError::Locked, StatusCode::FORBIDDEN),
            (MsrError::UnsupportedLayout, StatusCode::SERVICE_UNAVAILABLE),
            (
                MsrError::InvalidWindow("too large".into()),
                StatusCode::BAD_REQUEST,
            ),
        ] {
            let error = ZeusdError::cpu_msr(0, "set the time window", source);
            assert_eq!(error.status_code(), status, "{error}");
        }
    }
}
