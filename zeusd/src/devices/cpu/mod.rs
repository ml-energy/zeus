pub mod baseline;
pub mod hsmp;
pub mod msr;
mod rapl;
pub use rapl::{RaplCpu, HSMP_SOCKET_CONSTRAINT};

pub mod power;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::{Sender, UnboundedReceiver, UnboundedSender};
use tokio::time::{interval, Duration};
use tracing::Span;

use crate::error::{ZeusdError, PERMISSIONS_DOC_URL};
use hsmp::HSMP_DEVICE_PATH;
pub use msr::RaplPowerInfo;

pub struct PackageInfo {
    pub index: usize,
    pub name: String,
    /// The RAPL powercap zone directory.
    pub zone_dir: PathBuf,
    pub energy_uj_path: PathBuf,
    pub max_energy_uj: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RaplResponse {
    pub cpu_energy_uj: Option<u64>,
    pub dram_energy_uj: Option<u64>,
}

/// One power limit constraint of a CPU package or DRAM zone.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PowerLimitConstraint {
    /// `long_term`, `short_term`, or `peak_power` for RAPL constraints, as
    /// named by the kernel, or `socket` for the AMD HSMP socket power limit.
    pub name: String,
    pub power_limit_mw: u64,
    /// None if the kernel has no value for this constraint (its sysfs read fails
    /// with `ENODATA`).
    pub max_power_mw: Option<u64>,
    /// None if the kernel has no value for this constraint (its sysfs read fails
    /// with `ENODATA`), which is the case for `peak_power` on kernels 6.5 and later.
    /// Always None for `socket`, whose averaging window the firmware fixes.
    pub time_window_us: Option<u64>,
}

/// Power limit state of one CPU package or DRAM zone.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ZonePowerLimits {
    /// Whether the kernel reports the zone's `long_term` limit as enabled. The
    /// kernel reports false when the limit is disabled, when it is locked by the
    /// BIOS, or when reading its enable bit failed; sysfs does not distinguish
    /// these cases.
    pub enabled: bool,
    /// RAPL constraints in sysfs index order, followed by `socket` if the
    /// zone is a package with HSMP access. Empty if neither is available.
    pub constraints: Vec<PowerLimitConstraint>,
}

/// Power limit ranges a CPU package reports, per mechanism.
///
/// The hardware does not enforce these ranges: it can accept limits outside
/// them, and whether it holds a limit depends on the load.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CpuPowerLimitConstraints {
    /// From `MSR_PKG_POWER_INFO`, or None if the package zone has no RAPL constraints.
    pub rapl: Option<RaplPowerInfo>,
    /// From HSMP, or None if the package zone has no `socket` constraint.
    pub hsmp: Option<HsmpPowerInfo>,
}

/// Power limit range of the HSMP socket power limit.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HsmpPowerInfo {
    /// Largest socket power limit the firmware applies; it clamps higher ones.
    pub max_power_mw: u64,
}

/// Power limits of a CPU package zone and its DRAM zone.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CpuDramPowerLimits {
    pub cpu: ZonePowerLimits,
    pub dram: Option<ZonePowerLimits>,
}

/// Response from a CPU command.
#[derive(Debug)]
pub enum CpuResponse {
    Ok,
    Energy(RaplResponse),
    PowerLimits(CpuDramPowerLimits),
    PowerLimitConstraints(CpuPowerLimitConstraints),
}

pub trait CpuManager {
    /// Get the number of CPUs available.
    fn device_count() -> Result<usize, ZeusdError>;
    /// Get the CPU PackageInfo and the DRAM PackageInfo it is available.
    fn get_available_fields(
        index: usize,
    ) -> Result<(Arc<PackageInfo>, Option<Arc<PackageInfo>>), ZeusdError>;
    /// Get the cumulative energy counter value of the CPU after compensating for wraparounds.
    fn get_cpu_energy(&mut self) -> Result<u64, ZeusdError>;
    /// Get the cumulative energy counter value of the DRAM after compensating for wraparounds.
    fn get_dram_energy(&mut self) -> Result<u64, ZeusdError>;
    /// Check if DRAM is available.
    fn is_dram_available(&self) -> bool;
    /// Read the power limits of the CPU package zone and, if available, the DRAM zone.
    fn get_power_limits(&self) -> Result<CpuDramPowerLimits, ZeusdError>;
    /// Read the power limit ranges the CPU package reports.
    fn get_power_limit_constraints(&self) -> Result<CpuPowerLimitConstraints, ZeusdError>;
    /// Write the power limit of a package zone constraint in milliwatts.
    ///
    /// Callers check that the constraint exists and the limit is positive.
    /// Implementations reject limits the hardware would not apply as given.
    fn set_power_limit(&mut self, constraint: &str, power_limit_mw: u64) -> Result<(), ZeusdError>;
    /// Write the time window of a package zone constraint in microseconds.
    ///
    /// Callers validate the value against `get_power_limits` first.
    fn set_power_limit_time_window(
        &mut self,
        constraint: &str,
        time_window_us: u64,
    ) -> Result<(), ZeusdError>;
    /// Restore the power limits and time windows of the package zone
    /// constraints to their recorded baseline.
    fn reset_power_limits(&mut self) -> Result<(), ZeusdError>;
}

pub type CpuCommandRequest = (
    CpuCommand,
    Option<Sender<Result<CpuResponse, ZeusdError>>>,
    Instant,
    Span,
);

#[derive(Clone)]
pub struct CpuManagementTasks {
    // Senders to the CPU management tasks. index is the CPU ID.
    senders: Vec<UnboundedSender<CpuCommandRequest>>,
}

impl CpuManagementTasks {
    pub fn start<T>(cpus: Vec<T>) -> Result<Self, ZeusdError>
    where
        T: CpuManager + Send + 'static,
    {
        let mut senders = Vec::with_capacity(cpus.len());
        for (cpu_id, cpu) in cpus.into_iter().enumerate() {
            // Channel to send commands to the CPU management task.
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            senders.push(tx);
            // The CPU management task will automatically terminate
            // when the server terminates and the last sender is dropped.
            tokio::spawn(cpu_management_task(cpu, rx));
            tracing::info!("Background task for CPU {} successfully spawned", cpu_id);
        }
        Ok(Self { senders })
    }

    /// Return the number of CPUs managed by these tasks.
    pub fn device_count(&self) -> usize {
        self.senders.len()
    }

    pub async fn send_command_blocking(
        &self,
        cpu_id: usize,
        command: CpuCommand,
        request_start_time: Instant,
    ) -> Result<CpuResponse, ZeusdError> {
        if cpu_id >= self.senders.len() {
            return Err(ZeusdError::CpuNotFoundError(cpu_id));
        }
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        self.senders[cpu_id]
            .send((command, Some(tx), request_start_time, Span::current()))
            .map_err(ZeusdError::from)?;
        match rx.recv().await {
            Some(result) => result,
            None => Err(ZeusdError::CpuManagementTaskTerminatedError(cpu_id)),
        }
    }
}

/// A CPU command that can be executed on a CPU.
#[derive(Debug, Clone)]
pub enum CpuCommand {
    /// Get the CPU and DRAM energy measurement for the CPU index.
    GetIndexEnergy { cpu: bool, dram: bool },
    /// Get the power limits of the CPU package and DRAM zones.
    GetPowerLimits,
    /// Get the power limit ranges the CPU package reports.
    GetPowerLimitConstraints,
    /// Set the power limit of a package zone constraint.
    SetPowerLimit {
        constraint: String,
        power_limit_mw: u64,
    },
    /// Set the time window of a package zone constraint.
    SetPowerLimitTimeWindow {
        constraint: String,
        time_window_us: u64,
    },
    /// Restore the package zone constraints to their recorded baseline.
    ResetPowerLimits,
}

/// Tokio background task that handles requests to each CPU.
///
/// Between commands, a periodic keepalive reads the energy counters every 30
/// seconds so that RAPL counter wraparounds are detected even during idle
/// periods when no client is actively querying energy.
async fn cpu_management_task<T: CpuManager>(
    mut cpu: T,
    mut rx: UnboundedReceiver<CpuCommandRequest>,
) {
    let mut keepalive = interval(Duration::from_secs(30));
    // The first tick completes immediately; consume it so we don't
    // do a spurious keepalive read right at startup.
    keepalive.tick().await;

    loop {
        tokio::select! {
            Some((command, response, start_time, span)) = rx.recv() => {
                let _span_guard = span.enter();
                let result = command.execute(&mut cpu, start_time);
                if let Err(e) = &result {
                    tracing::warn!("CPU command {command:?} failed: {e}");
                }
                if let Some(response) = response {
                    if response.send(result).await.is_err() {
                        tracing::error!("Failed to send response to caller");
                    }
                }
            }
            _ = keepalive.tick() => {
                let _ = cpu.get_cpu_energy();
                if cpu.is_dram_available() {
                    let _ = cpu.get_dram_energy();
                }
            }
            else => break,
        }
    }
}

impl CpuCommand {
    fn execute<T>(
        &self,
        device: &mut T,
        _request_arrival_time: Instant,
    ) -> Result<CpuResponse, ZeusdError>
    where
        T: CpuManager,
    {
        match self {
            &Self::GetIndexEnergy { cpu, dram } => {
                let cpu_energy_uj = if cpu {
                    Some(device.get_cpu_energy()?)
                } else {
                    None
                };
                let dram_energy_uj = if dram && device.is_dram_available() {
                    Some(device.get_dram_energy()?)
                } else {
                    None
                };
                Ok(CpuResponse::Energy(RaplResponse {
                    cpu_energy_uj,
                    dram_energy_uj,
                }))
            }
            Self::GetPowerLimits => device.get_power_limits().map(CpuResponse::PowerLimits),
            Self::GetPowerLimitConstraints => device
                .get_power_limit_constraints()
                .map(CpuResponse::PowerLimitConstraints),
            Self::SetPowerLimit {
                constraint,
                power_limit_mw,
            } => {
                let limits = device.get_power_limits()?;
                validate_power_limit(&limits.cpu, constraint, *power_limit_mw)?;
                device.set_power_limit(constraint, *power_limit_mw)?;
                Ok(CpuResponse::Ok)
            }
            Self::SetPowerLimitTimeWindow {
                constraint,
                time_window_us,
            } => {
                let limits = device.get_power_limits()?;
                validate_time_window(&limits.cpu, constraint, *time_window_us)?;
                device.set_power_limit_time_window(constraint, *time_window_us)?;
                Ok(CpuResponse::Ok)
            }
            Self::ResetPowerLimits => {
                device.reset_power_limits()?;
                Ok(CpuResponse::Ok)
            }
        }
    }
}

fn find_constraint<'a>(
    zone: &'a ZonePowerLimits,
    constraint: &str,
) -> Result<&'a PowerLimitConstraint, ZeusdError> {
    zone.constraints
        .iter()
        .find(|c| c.name == constraint)
        .ok_or_else(|| {
            let available: Vec<&str> = zone.constraints.iter().map(|c| c.name.as_str()).collect();
            let hint = if constraint == HSMP_SOCKET_CONSTRAINT {
                format!(
                    " The '{HSMP_SOCKET_CONSTRAINT}' constraint needs AMD HSMP: load the amd_hsmp \
                     kernel module so that {HSMP_DEVICE_PATH} exists (in a container, also pass \
                     the device) and restart Zeusd. See {PERMISSIONS_DOC_URL}"
                )
            } else {
                String::new()
            };
            ZeusdError::InvalidRequest(format!(
                "Package zone has no power limit constraint '{constraint}' (available: {available:?}).{hint}"
            ))
        })
}

/// Reject a power limit of zero or for a constraint the zone does not have.
///
/// Neither RAPL nor HSMP reports a minimum the hardware can enforce, so zero is
/// the only value rejected for being too low. Upper bounds depend on how the
/// constraint is applied, so `CpuManager::set_power_limit` checks them.
fn validate_power_limit(
    zone: &ZonePowerLimits,
    constraint: &str,
    power_limit_mw: u64,
) -> Result<(), ZeusdError> {
    find_constraint(zone, constraint)?;
    if power_limit_mw == 0 {
        return Err(ZeusdError::InvalidRequest(
            "Power limit must be positive".to_string(),
        ));
    }
    Ok(())
}

/// Reject a time window of zero or one for a constraint without a time window.
fn validate_time_window(
    zone: &ZonePowerLimits,
    constraint: &str,
    time_window_us: u64,
) -> Result<(), ZeusdError> {
    let found = find_constraint(zone, constraint)?;
    if found.time_window_us.is_none() {
        return Err(ZeusdError::InvalidRequest(format!(
            "Constraint '{constraint}' has no adjustable time window"
        )));
    }
    if time_window_us == 0 {
        return Err(ZeusdError::InvalidRequest(
            "Time window must be positive".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zone(max_power_mw: Option<u64>) -> ZonePowerLimits {
        ZonePowerLimits {
            enabled: true,
            constraints: vec![PowerLimitConstraint {
                name: "long_term".to_string(),
                power_limit_mw: 100_000,
                max_power_mw,
                time_window_us: Some(999_424),
            }],
        }
    }

    #[test]
    fn power_limit_must_be_positive_for_an_existing_constraint() {
        assert!(validate_power_limit(&zone(Some(205_000)), "long_term", 250_000).is_ok());
        assert!(validate_power_limit(&zone(None), "long_term", 1).is_ok());
        assert!(validate_power_limit(&zone(None), "long_term", 0).is_err());
        assert!(validate_power_limit(&zone(None), "short_term", 1).is_err());
    }

    #[test]
    fn missing_socket_constraint_explains_hsmp_prerequisite() {
        let message = validate_power_limit(&zone(None), "socket", 1)
            .unwrap_err()
            .to_string();
        assert!(message.contains("amd_hsmp"), "{message}");
        assert!(message.contains(PERMISSIONS_DOC_URL), "{message}");

        let message = validate_power_limit(&zone(None), "short_term", 1)
            .unwrap_err()
            .to_string();
        assert!(!message.contains("amd_hsmp"), "{message}");
    }
}
