//! CPU power measurement with RAPL and power capping with RAPL and AMD HSMP.
//! Only supported on Linux.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::string::String;
use std::sync::Arc;

use once_cell::sync::Lazy;

use crate::devices::cpu::hsmp::{HsmpError, HsmpSocket, HsmpTransport, HSMP_DEVICE_PATH};
use crate::devices::cpu::msr::{parse_package_zone_name, read_power_info, TimeWindows};
use crate::devices::cpu::power_limit_snapshot::{ConstraintSetting, PackageSettings};
use crate::devices::cpu::{
    missing_constraint_error, CpuDramPowerLimits, CpuManager, CpuPowerLimitConstraints,
    HsmpPowerInfo, PackageInfo, PowerLimitConstraint, ZonePowerLimits,
};
use crate::error::{ZeusdError, PERMISSIONS_DOC_URL, RAPL_AVAILABILITY};

/// Name of the package zone constraint backed by the AMD HSMP socket power limit.
pub const HSMP_SOCKET_CONSTRAINT: &str = "socket";

static SYS_RAPL_DIR: &str = "/sys/class/powercap/intel-rapl";

// Docker masks `/sys/devices/virtual/powercap` by default, so containerized
// deployments bind-mount the host's RAPL directories under `/zeus_sys` instead.
// Same convention as the Zeus Python package.
static CONTAINER_RAPL_DIR: &str = "/zeus_sys/class/powercap/intel-rapl";

/// CPU topology, used to find a CPU of a package for reading its MSRs.
static SYS_CPU_DIR: &str = "/sys/devices/system/cpu";
/// Per-CPU MSR devices created by the `msr` kernel module.
static DEV_CPU_DIR: &str = "/dev/cpu";

static RAPL_DIR: Lazy<&'static str> = Lazy::new(|| {
    if Path::new(CONTAINER_RAPL_DIR).exists() {
        tracing::info!("Reading RAPL through the container mount at {CONTAINER_RAPL_DIR}");
        CONTAINER_RAPL_DIR
    } else {
        SYS_RAPL_DIR
    }
});

pub struct RaplCpu {
    cpu: Arc<PackageInfo>,
    dram: Option<Arc<PackageInfo>>,
    last_cpu_raw_uj: Option<u64>,
    cpu_wraparound_count: u64,
    last_dram_raw_uj: Option<u64>,
    dram_wraparound_count: u64,
    /// HSMP access to this package's socket, if attached.
    hsmp: Option<HsmpSocket>,
    /// Original package zone constraint settings that `reset_power_limits` restores.
    original_power_limits: Option<Vec<ConstraintSetting>>,
}

/// Where a package zone constraint is read and written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstraintTarget {
    /// The `constraint_<index>_*` files of the powercap zone.
    Powercap(usize),
    /// The HSMP socket power limit.
    HsmpSocket,
}

/// A package zone constraint in the units of the kernel's powercap interface.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RawConstraint {
    target: ConstraintTarget,
    name: String,
    power_limit_uw: u64,
    max_power_uw: Option<u64>,
    time_window_us: Option<u64>,
}

impl RawConstraint {
    fn to_constraint(&self) -> PowerLimitConstraint {
        PowerLimitConstraint {
            name: self.name.clone(),
            power_limit_mw: self.power_limit_uw / 1000,
            max_power_mw: self.max_power_uw.map(|uw| uw / 1000),
            time_window_us: self.time_window_us,
        }
    }
}

impl RaplCpu {
    pub fn init(index: usize) -> Result<Self, ZeusdError> {
        let fields = RaplCpu::get_available_fields(index)?;
        Ok(Self {
            cpu: fields.0,
            dram: fields.1,
            last_cpu_raw_uj: None,
            cpu_wraparound_count: 0,
            last_dram_raw_uj: None,
            dram_wraparound_count: 0,
            hsmp: None,
            original_power_limits: None,
        })
    }

    /// Name of the package zone, such as `package-0`.
    pub fn zone_name(&self) -> &str {
        &self.cpu.name
    }

    /// Expose the HSMP socket power limit of this package as the `socket` constraint.
    ///
    /// The kernel names RAPL package zones `package-<N>` after the physical
    /// package ID, which is the HSMP socket index on AMD EPYC.
    pub fn attach_hsmp(&mut self, transport: Arc<dyn HsmpTransport>) -> Result<(), ZeusdError> {
        let sock_ind = self
            .cpu
            .name
            .strip_prefix("package-")
            .and_then(|id| id.parse::<u16>().ok())
            .ok_or_else(|| {
                ZeusdError::InvalidRequest(format!(
                    "Cannot map RAPL zone '{}' of CPU {} to an HSMP socket; expected a name of the form 'package-<N>'",
                    self.cpu.name, self.cpu.index
                ))
            })?;
        self.hsmp = Some(HsmpSocket::new(transport, sock_ind));
        Ok(())
    }

    /// Read the current power limit and time window of every package zone
    /// constraint to record them as the original settings.
    ///
    /// HSMP is optional, so a failed HSMP read leaves `socket` out of the
    /// settings and is returned next to them instead of failing the read.
    pub fn power_limit_settings(
        &self,
    ) -> Result<(PackageSettings, Option<ZeusdError>), ZeusdError> {
        let mut constraints = read_raw_constraints(&self.cpu.zone_dir)?;
        let mut hsmp_error = None;
        if let Some(hsmp) = &self.hsmp {
            match self.socket_constraint(hsmp) {
                Ok(socket) => constraints.push(socket),
                Err(error) => hsmp_error = Some(error),
            }
        }
        let settings = PackageSettings {
            zone: self.cpu.name.clone(),
            constraints: constraints
                .into_iter()
                .map(|c| ConstraintSetting {
                    name: c.name,
                    power_limit_uw: c.power_limit_uw,
                    time_window_us: c.time_window_us,
                })
                .collect(),
        };
        Ok((settings, hsmp_error))
    }

    /// Set the original settings that `reset_power_limits` restores.
    ///
    /// Only constraints that have an original setting can be changed.
    pub fn set_original_power_limits(&mut self, constraints: Vec<ConstraintSetting>) {
        self.original_power_limits = Some(constraints);
    }

    /// Read the RAPL constraints of the package zone followed by the HSMP
    /// `socket` constraint, if attached.
    fn package_constraints(&self) -> Result<Vec<RawConstraint>, ZeusdError> {
        let mut constraints = read_raw_constraints(&self.cpu.zone_dir)?;
        if let Some(hsmp) = &self.hsmp {
            constraints.push(self.socket_constraint(hsmp)?);
        }
        Ok(constraints)
    }

    /// Read the HSMP `socket` constraint.
    fn socket_constraint(&self, hsmp: &HsmpSocket) -> Result<RawConstraint, ZeusdError> {
        Ok(RawConstraint {
            target: ConstraintTarget::HsmpSocket,
            name: HSMP_SOCKET_CONSTRAINT.to_string(),
            power_limit_uw: u64::from(
                self.read_hsmp(hsmp.power_limit_mw(), "read the HSMP socket power limit")?,
            ) * 1000,
            max_power_uw: Some(self.read_hsmp_max_power_uw(hsmp)?),
            time_window_us: None,
        })
    }

    /// Error unless `constraint` has an original setting that reset can restore.
    fn check_original(&self, constraint: &str) -> Result<(), ZeusdError> {
        let original = self
            .original_power_limits
            .as_ref()
            .ok_or_else(|| ZeusdError::CpuOriginalPowerLimitsMissingError(self.cpu.index))?;
        if original.iter().any(|setting| setting.name == constraint) {
            Ok(())
        } else {
            Err(ZeusdError::CpuConstraintOriginalMissingError {
                cpu: self.cpu.index,
                constraint: constraint.to_string(),
            })
        }
    }

    fn read_hsmp_max_power_uw(&self, hsmp: &HsmpSocket) -> Result<u64, ZeusdError> {
        let max_mw = self.read_hsmp(
            hsmp.power_limit_max_mw(),
            "read the HSMP maximum socket power limit",
        )?;
        Ok(u64::from(max_mw) * 1000)
    }

    fn read_hsmp<T>(
        &self,
        result: Result<T, HsmpError>,
        action: &'static str,
    ) -> Result<T, ZeusdError> {
        result.map_err(|source| ZeusdError::cpu_hsmp(self.cpu.index, action, false, source))
    }

    /// Return an error for each energy counter that cannot be read, without
    /// changing the wraparound state.
    pub fn check_energy_access(&self) -> Vec<ZeusdError> {
        std::iter::once(&self.cpu)
            .chain(&self.dram)
            .filter_map(|zone| read_energy_uj(zone).err())
            .collect()
    }

    /// Read one package zone constraint through only the interface that provides it.
    fn find_package_constraint(&self, constraint: &str) -> Result<RawConstraint, ZeusdError> {
        if let (HSMP_SOCKET_CONSTRAINT, Some(hsmp)) = (constraint, &self.hsmp) {
            return self.socket_constraint(hsmp);
        }
        let rapl = read_raw_constraints(&self.cpu.zone_dir)?;
        if let Some(found) = rapl.iter().find(|c| c.name == constraint) {
            return Ok(found.clone());
        }
        let mut available: Vec<&str> = rapl.iter().map(|c| c.name.as_str()).collect();
        if self.hsmp.is_some() {
            available.push(HSMP_SOCKET_CONSTRAINT);
        }
        Err(missing_constraint_error(constraint, &available))
    }

    fn write_power_limit_uw(
        &self,
        target: ConstraintTarget,
        name: &str,
        power_limit_uw: u64,
    ) -> Result<(), ZeusdError> {
        let action = || {
            format!(
                "set the power limit of constraint '{name}' on CPU {} to {power_limit_uw} uW",
                self.cpu.index
            )
        };
        match target {
            ConstraintTarget::Powercap(index) => {
                let path = self
                    .cpu
                    .zone_dir
                    .join(format!("constraint_{index}_power_limit_uw"));
                fs::write(path, power_limit_uw.to_string())
                    .map_err(|source| ZeusdError::cpu_control(action(), source))
            }
            ConstraintTarget::HsmpSocket => {
                let hsmp = self.hsmp.as_ref().ok_or_else(|| {
                    ZeusdError::InvalidRequest(format!("CPU {} has no HSMP access", self.cpu.index))
                })?;
                let power_limit_mw = u32::try_from(power_limit_uw / 1000).map_err(|_| {
                    ZeusdError::InvalidRequest(format!(
                        "Power limit {power_limit_uw} uW is out of range for HSMP"
                    ))
                })?;
                hsmp.set_power_limit_mw(power_limit_mw)
                    .map_err(|source| ZeusdError::cpu_hsmp(self.cpu.index, action(), true, source))
            }
        }
    }

    fn write_time_window_us(
        &self,
        target: ConstraintTarget,
        name: &str,
        time_window_us: u64,
        exact: bool,
    ) -> Result<(), ZeusdError> {
        if !matches!(target, ConstraintTarget::Powercap(_)) {
            return Err(ZeusdError::InvalidRequest(format!(
                "Constraint '{name}' has no adjustable time window"
            )));
        }
        let action = if exact {
            "restore the time window exactly"
        } else {
            "set the time window"
        };
        let (package, die) = parse_package_zone_name(&self.cpu.name).ok_or_else(|| {
            ZeusdError::InvalidRequest(format!(
                "Zone '{}' has no package MSR time-window control",
                self.cpu.name
            ))
        })?;
        let windows = TimeWindows::open(
            Path::new(SYS_CPU_DIR),
            Path::new(DEV_CPU_DIR),
            package,
            die,
            true,
        )
        .map_err(|source| ZeusdError::cpu_msr(self.cpu.index, action, source))?;
        windows
            .set(name, time_window_us, exact)
            .map_err(|source| ZeusdError::cpu_msr(self.cpu.index, action, source))
    }

    /// Log Intel MSR availability without changing registers or requiring MSR for monitoring.
    pub fn log_msr_availability(&self, control_enabled: bool) {
        match read_raw_constraints(&self.cpu.zone_dir) {
            Ok(constraints) if constraints.is_empty() => return,
            Ok(_) => {}
            Err(error) => {
                tracing::warn!(
                    cpu = self.cpu.index,
                    "Power limit queries and control of CPU {} will fail: {error}",
                    self.cpu.index,
                );
                return;
            }
        }
        let Some((package, die)) = parse_package_zone_name(&self.cpu.name) else {
            return;
        };
        match read_power_info(Path::new(SYS_CPU_DIR), Path::new(DEV_CPU_DIR), package, die) {
            Ok(_) => tracing::info!(
                cpu = self.cpu.index,
                "Intel hardware-range queries have MSR read access"
            ),
            Err(source) => {
                ZeusdError::cpu_msr(self.cpu.index, "read Intel hardware ranges", source);
            }
        }
        if control_enabled {
            match TimeWindows::open(Path::new(SYS_CPU_DIR), Path::new(DEV_CPU_DIR), package, die, true) {
                Ok(_) => tracing::info!(cpu = self.cpu.index,
                    "MSR device opened for time-window control. Writes are checked on each request; kernel lockdown, msr.allow_writes, and BIOS locks can reject them. Userspace MSR writes taint the kernel until reboot."),
                Err(source) => { ZeusdError::cpu_msr(self.cpu.index, "open Intel time-window control", source); }
            }
        }
    }
}

impl PackageInfo {
    /// Read the zone at `base_path`.
    ///
    /// The energy counter is not read, so zones whose energy counter Zeusd
    /// cannot read still support power limit queries and control.
    pub fn new(base_path: &Path, index: usize) -> anyhow::Result<Self, ZeusdError> {
        let cpu_name_path = base_path.join("name");
        let cpu_energy_path = base_path.join("energy_uj");
        let cpu_max_energy_path = base_path.join("max_energy_range_uj");

        let init_error = |reason: String| ZeusdError::CpuInitializationError { cpu: index, reason };
        for path in [&cpu_name_path, &cpu_max_energy_path, &cpu_energy_path] {
            match path.try_exists() {
                Ok(true) => {}
                Ok(false) => return Err(init_error(format!("{} does not exist", path.display()))),
                Err(e) => return Err(init_error(format!("cannot access {}: {e}", path.display()))),
            }
        }

        let cpu_name = fs::read_to_string(&cpu_name_path)
            .map_err(|e| init_error(format!("cannot read {}: {e}", cpu_name_path.display())))?
            .trim_end()
            .to_string();
        let cpu_max_energy = read_u64(&cpu_max_energy_path).map_err(|e| {
            init_error(format!(
                "cannot read {}: {e}",
                cpu_max_energy_path.display()
            ))
        })?;
        Ok(PackageInfo {
            index,
            name: cpu_name,
            zone_dir: base_path.to_path_buf(),
            energy_uj_path: cpu_energy_path,
            max_energy_uj: cpu_max_energy,
        })
    }
}

impl CpuManager for RaplCpu {
    fn device_count() -> Result<usize, ZeusdError> {
        let mut index_count = 0;
        let base_path = PathBuf::from(*RAPL_DIR);

        match fs::read_dir(&base_path) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        if let Some(dir_name_str) = path.file_name() {
                            let dir_name = dir_name_str.to_string_lossy();
                            if dir_name.contains("intel-rapl") {
                                index_count += 1;
                            }
                        }
                    }
                }
            }
            Err(e) => {
                tracing::error!(
                    "RAPL is not available: cannot read {}: {e}. CPU energy monitoring and RAPL \
                     power limits need it. {RAPL_AVAILABILITY} See {PERMISSIONS_DOC_URL}",
                    base_path.display(),
                );
            }
        };
        Ok(index_count)
    }

    fn get_available_fields(
        index: usize,
    ) -> Result<(Arc<PackageInfo>, Option<Arc<PackageInfo>>), ZeusdError> {
        let base_path = PathBuf::from(format!("{}/intel-rapl:{index}", *RAPL_DIR));
        let cpu_info = PackageInfo::new(&base_path, index)?;

        match fs::read_dir(&base_path) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        if let Some(dir_name_str) = path.file_name() {
                            let dir_name = dir_name_str.to_string_lossy();
                            if dir_name.contains("intel-rapl") {
                                let subpackage_path = base_path.join(&*dir_name);
                                let subpackage_info = PackageInfo::new(&subpackage_path, index)?;
                                if subpackage_info.name == "dram" {
                                    return Ok((
                                        Arc::new(cpu_info),
                                        Some(Arc::new(subpackage_info)),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                return Err(ZeusdError::CpuInitializationError {
                    cpu: index,
                    reason: format!("cannot read {}: {e}", base_path.display()),
                });
            }
        };

        Ok((Arc::new(cpu_info), None))
    }

    fn get_cpu_energy(&mut self) -> Result<u64, ZeusdError> {
        let raw = read_energy_uj(&self.cpu)?;
        if let Some(last_raw) = self.last_cpu_raw_uj {
            if raw < last_raw {
                self.cpu_wraparound_count += 1;
            }
        }
        self.last_cpu_raw_uj = Some(raw);
        Ok(raw + self.cpu_wraparound_count * self.cpu.max_energy_uj)
    }

    fn get_dram_energy(&mut self) -> Result<u64, ZeusdError> {
        match &self.dram {
            None => Err(ZeusdError::CpuManagementTaskTerminatedError(self.cpu.index)),
            Some(dram) => {
                let raw = read_energy_uj(dram)?;
                if let Some(last_raw) = self.last_dram_raw_uj {
                    if raw < last_raw {
                        self.dram_wraparound_count += 1;
                    }
                }
                self.last_dram_raw_uj = Some(raw);
                Ok(raw + self.dram_wraparound_count * dram.max_energy_uj)
            }
        }
    }

    fn is_dram_available(&self) -> bool {
        self.dram.is_some()
    }

    fn get_power_limits(&self) -> Result<CpuDramPowerLimits, ZeusdError> {
        Ok(CpuDramPowerLimits {
            cpu: ZonePowerLimits {
                enabled: read_zone_enabled(&self.cpu.zone_dir)?,
                constraints: self
                    .package_constraints()?
                    .iter()
                    .map(RawConstraint::to_constraint)
                    .collect(),
            },
            dram: self
                .dram
                .as_ref()
                .map(|dram| read_zone_limits(&dram.zone_dir))
                .transpose()?,
        })
    }

    fn get_power_limit_constraints(&self) -> Result<CpuPowerLimitConstraints, ZeusdError> {
        // Platform domains such as psys have no package power-info register.
        let package = parse_package_zone_name(&self.cpu.name);
        let rapl = match package {
            Some((package_id, die_id)) if !read_raw_constraints(&self.cpu.zone_dir)?.is_empty() => {
                Some(
                    read_power_info(
                        Path::new(SYS_CPU_DIR),
                        Path::new(DEV_CPU_DIR),
                        package_id,
                        die_id,
                    )
                    .map_err(|source| {
                        ZeusdError::cpu_msr(self.cpu.index, "read Intel hardware ranges", source)
                    })?,
                )
            }
            _ => None,
        };
        let hsmp = match &self.hsmp {
            Some(hsmp) => Some(HsmpPowerInfo {
                max_power_mw: self.read_hsmp_max_power_uw(hsmp)? / 1000,
            }),
            None => None,
        };
        Ok(CpuPowerLimitConstraints { rapl, hsmp })
    }

    fn get_package_constraint(&self, constraint: &str) -> Result<PowerLimitConstraint, ZeusdError> {
        Ok(self.find_package_constraint(constraint)?.to_constraint())
    }

    /// Reject an HSMP limit above the firmware's maximum, which the firmware
    /// would clamp, and a RAPL limit that does not fit the register field.
    ///
    /// The register's power unit and field width are not exposed in sysfs, so
    /// a RAPL limit is written and read back instead. If it did not fit, the
    /// previous limit is written back before rejecting the request.
    fn set_power_limit(&mut self, constraint: &str, power_limit_mw: u64) -> Result<(), ZeusdError> {
        let found = self.find_package_constraint(constraint)?;
        self.check_original(&found.name)?;
        let power_limit_uw = power_limit_mw.checked_mul(1000).ok_or_else(|| {
            ZeusdError::InvalidRequest(format!("Power limit {power_limit_mw} mW is out of range"))
        })?;
        match found.target {
            ConstraintTarget::HsmpSocket => {
                if let Some(max_power_uw) = found.max_power_uw {
                    if power_limit_uw > max_power_uw {
                        return Err(ZeusdError::InvalidRequest(format!(
                            "Power limit {power_limit_mw} mW exceeds the maximum {} mW of constraint '{}'",
                            max_power_uw / 1000,
                            found.name
                        )));
                    }
                }
                self.write_power_limit_uw(found.target, &found.name, power_limit_uw)
            }
            ConstraintTarget::Powercap(index) => {
                self.write_power_limit_uw(found.target, &found.name, power_limit_uw)?;
                let stored_path = self
                    .cpu
                    .zone_dir
                    .join(format!("constraint_{index}_power_limit_uw"));
                let stored_uw = read_u64(&stored_path)
                    .map_err(|source| limit_read_error(&stored_path, source))?;
                if stored_as_requested(power_limit_uw, stored_uw) {
                    return Ok(());
                }
                let rejection = ZeusdError::InvalidRequest(format!(
                    "Power limit {power_limit_mw} mW does not fit the RAPL register of constraint '{}' \
                     (the kernel stored {} mW); the previous limit was restored",
                    found.name,
                    stored_uw / 1000
                ));
                match self.write_power_limit_uw(found.target, &found.name, found.power_limit_uw) {
                    Ok(()) => Err(rejection),
                    Err(restore_error) => Err(ZeusdError::Multiple(vec![rejection, restore_error])),
                }
            }
        }
    }

    fn set_power_limit_time_window(
        &mut self,
        constraint: &str,
        time_window_us: u64,
    ) -> Result<(), ZeusdError> {
        let found = self.find_package_constraint(constraint)?;
        self.check_original(&found.name)?;
        if found.time_window_us.is_none() || time_window_us == 0 {
            return Err(ZeusdError::InvalidRequest(format!(
                "Constraint '{constraint}' must have an adjustable time window, and time_window_us must be positive"
            )));
        }
        self.write_time_window_us(found.target, &found.name, time_window_us, false)
    }

    /// Restore each original setting that differs from the current one.
    ///
    /// Settings that already match are not written, so constraints the BIOS
    /// locked, which can never differ from their original settings, do not fail
    /// the reset. RAPL and HSMP are read separately, so a failed read of one
    /// does not stop restoring the other.
    fn reset_power_limits(&mut self) -> Result<(), ZeusdError> {
        let original = self
            .original_power_limits
            .as_ref()
            .ok_or_else(|| ZeusdError::CpuOriginalPowerLimitsMissingError(self.cpu.index))?;
        // Attempt every read and write so one failure does not leave the rest unrestored.
        let mut errors = Vec::new();
        let rapl = match read_raw_constraints(&self.cpu.zone_dir) {
            Ok(rapl) => Some(rapl),
            Err(error) => {
                errors.push(error);
                None
            }
        };
        for setting in original {
            let found = if setting.name == HSMP_SOCKET_CONSTRAINT {
                let socket = match &self.hsmp {
                    Some(hsmp) => self.socket_constraint(hsmp),
                    None => Err(ZeusdError::cpu_hsmp(
                        self.cpu.index,
                        "read the HSMP socket power limit",
                        false,
                        HsmpError::DeviceMissing(PathBuf::from(HSMP_DEVICE_PATH)),
                    )),
                };
                match socket {
                    Ok(socket) => socket,
                    Err(error) => {
                        errors.push(error);
                        continue;
                    }
                }
            } else {
                // A failed RAPL read was reported above.
                let Some(rapl) = &rapl else { continue };
                match rapl.iter().find(|c| c.name == setting.name) {
                    Some(found) => found.clone(),
                    None => {
                        errors.push(ZeusdError::IOError(std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            format!(
                                "Constraint '{}' of CPU {} disappeared after its original setting was recorded",
                                setting.name, self.cpu.index
                            ),
                        )));
                        continue;
                    }
                }
            };
            if found.power_limit_uw != setting.power_limit_uw {
                if let Err(e) =
                    self.write_power_limit_uw(found.target, &found.name, setting.power_limit_uw)
                {
                    errors.push(e);
                }
            }
            if let Some(time_window_us) = setting.time_window_us {
                if found.time_window_us != Some(time_window_us) {
                    if let Err(e) =
                        self.write_time_window_us(found.target, &found.name, time_window_us, true)
                    {
                        errors.push(e);
                    }
                }
            }
        }
        ZeusdError::from_errors(errors)
    }
}

/// Whether the kernel stored a RAPL power limit as requested.
///
/// The kernel rounds a limit down to a multiple of the CPU's power unit, which
/// is `1_000_000 >> n` uW, so rounding loses less than 1 W. It also silently
/// drops bits that do not fit the register field, which lowers the limit by at
/// least 2^15 power units, i.e., at least 1 W even for the finest unit.
fn stored_as_requested(requested_uw: u64, stored_uw: u64) -> bool {
    stored_uw <= requested_uw && requested_uw - stored_uw < 1_000_000
}

/// Read the power limit state of a RAPL powercap zone.
fn read_zone_limits(zone_dir: &Path) -> Result<ZonePowerLimits, ZeusdError> {
    read_zone_limits_with(zone_dir, |path| fs::read_to_string(path))
}

/// [`read_zone_limits`] with an injectable file reader.
fn read_zone_limits_with(
    zone_dir: &Path,
    read_file: impl Fn(&Path) -> std::io::Result<String>,
) -> Result<ZonePowerLimits, ZeusdError> {
    Ok(ZonePowerLimits {
        enabled: read_zone_enabled_with(zone_dir, &read_file)?,
        constraints: read_raw_constraints_with(zone_dir, &read_file)?
            .iter()
            .map(RawConstraint::to_constraint)
            .collect(),
    })
}

fn read_zone_enabled(zone_dir: &Path) -> Result<bool, ZeusdError> {
    read_zone_enabled_with(zone_dir, |path| fs::read_to_string(path))
}

fn read_zone_enabled_with(
    zone_dir: &Path,
    read_file: impl Fn(&Path) -> std::io::Result<String>,
) -> Result<bool, ZeusdError> {
    let path = zone_dir.join("enabled");
    match parse_u64_with(&path, &read_file).map_err(|source| limit_read_error(&path, source))? {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(limit_read_error(
            &path,
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unexpected value {value}"),
            ),
        )),
    }
}

fn read_raw_constraints(zone_dir: &Path) -> Result<Vec<RawConstraint>, ZeusdError> {
    read_raw_constraints_with(zone_dir, |path| fs::read_to_string(path))
}

/// Read the constraints of a RAPL powercap zone.
///
/// The kernel numbers constraints from zero without gaps, so reading stops at
/// the first index whose `constraint_K_name` file does not exist. Any other
/// failure to read that file, such as missing read permission, is an error.
/// A constraint attribute the kernel has no value for fails with `ENODATA`
/// (e.g., the time window of `peak_power`) and is read as `None`.
fn read_raw_constraints_with(
    zone_dir: &Path,
    read_file: impl Fn(&Path) -> std::io::Result<String>,
) -> Result<Vec<RawConstraint>, ZeusdError> {
    let parse_u64 = |path: &Path| -> Result<u64, ZeusdError> {
        parse_u64_with(path, &read_file).map_err(|source| limit_read_error(path, source))
    };
    let parse_optional_u64 = |path: &Path| -> Result<Option<u64>, ZeusdError> {
        match parse_u64_with(path, &read_file) {
            Ok(value) => Ok(Some(value)),
            Err(e) if is_enodata(&e) => Ok(None),
            Err(source) => Err(limit_read_error(path, source)),
        }
    };

    let mut constraints = Vec::new();
    for index in 0.. {
        let file = |field: &str| zone_dir.join(format!("constraint_{index}_{field}"));
        let name_path = file("name");
        let name = match read_file(&name_path) {
            Ok(name) => name.trim_end().to_string(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => break,
            Err(source) => return Err(limit_read_error(&name_path, source)),
        };
        constraints.push(RawConstraint {
            target: ConstraintTarget::Powercap(index),
            name,
            power_limit_uw: parse_u64(&file("power_limit_uw"))?,
            max_power_uw: parse_optional_u64(&file("max_power_uw"))?,
            time_window_us: parse_optional_u64(&file("time_window_us"))?,
        });
    }
    Ok(constraints)
}

fn limit_read_error(path: &Path, source: std::io::Error) -> ZeusdError {
    ZeusdError::CpuLimitReadError {
        path: path.to_path_buf(),
        source,
    }
}

fn parse_u64_with(
    path: &Path,
    read_file: impl Fn(&Path) -> std::io::Result<String>,
) -> std::io::Result<u64> {
    read_file(path)?
        .trim()
        .parse()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn is_enodata(error: &std::io::Error) -> bool {
    #[cfg(unix)]
    {
        error.raw_os_error() == Some(nix::errno::Errno::ENODATA as i32)
    }
    #[cfg(not(unix))]
    {
        let _ = error;
        false
    }
}

fn read_energy_uj(zone: &PackageInfo) -> Result<u64, ZeusdError> {
    read_u64(&zone.energy_uj_path).map_err(|source| ZeusdError::CpuEnergyReadError {
        cpu: zone.index,
        path: zone.energy_uj_path.clone(),
        source,
    })
}

fn read_u64(path: &PathBuf) -> anyhow::Result<u64, std::io::Error> {
    let mut file = std::fs::File::open(path)?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;
    buf.trim()
        .parse()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::cpu::hsmp::tests::FakeHsmp;
    use crate::devices::cpu::CpuCommand;
    use std::path::Path;

    /// Write a u64 value to a file, simulating a RAPL energy counter.
    fn write_energy(path: &Path, value: u64) {
        fs::write(path, format!("{value}\n")).unwrap();
    }

    /// Create a RaplCpu backed by temp files with the given max_energy_uj.
    /// Returns the RaplCpu and the path to the CPU energy file.
    /// If `with_dram` is true, also creates a DRAM energy file.
    fn make_test_cpu(
        dir: &Path,
        max_energy_uj: u64,
        with_dram: bool,
    ) -> (RaplCpu, PathBuf, Option<PathBuf>) {
        let cpu_energy_path = dir.join("cpu_energy_uj");
        write_energy(&cpu_energy_path, 0);

        let cpu_info = Arc::new(PackageInfo {
            index: 0,
            name: "package-0".to_string(),
            zone_dir: dir.to_path_buf(),
            energy_uj_path: cpu_energy_path.clone(),
            max_energy_uj,
        });

        let (dram, dram_path) = if with_dram {
            let dram_energy_path = dir.join("dram_energy_uj");
            write_energy(&dram_energy_path, 0);
            let dram_info = Arc::new(PackageInfo {
                index: 0,
                name: "dram".to_string(),
                zone_dir: dir.to_path_buf(),
                energy_uj_path: dram_energy_path.clone(),
                max_energy_uj,
            });
            (Some(dram_info), Some(dram_energy_path))
        } else {
            (None, None)
        };

        let cpu = RaplCpu {
            cpu: cpu_info,
            dram,
            last_cpu_raw_uj: None,
            cpu_wraparound_count: 0,
            last_dram_raw_uj: None,
            dram_wraparound_count: 0,
            hsmp: None,
            original_power_limits: None,
        };

        (cpu, cpu_energy_path, dram_path)
    }

    #[test]
    fn monotonic_increase_no_wraparound() {
        let tmp = tempfile::tempdir().unwrap();
        let (mut cpu, path, _) = make_test_cpu(tmp.path(), 1_000_000, false);

        let values = [100, 500, 1_000, 50_000, 999_999];
        for &v in &values {
            write_energy(&path, v);
            assert_eq!(cpu.get_cpu_energy().unwrap(), v);
        }
    }

    #[test]
    fn single_wraparound() {
        let tmp = tempfile::tempdir().unwrap();
        let max = 1_000_000;
        let (mut cpu, path, _) = make_test_cpu(tmp.path(), max, false);

        // Counter climbs to near max.
        write_energy(&path, 900_000);
        assert_eq!(cpu.get_cpu_energy().unwrap(), 900_000);

        // Counter wraps around.
        write_energy(&path, 100);
        assert_eq!(cpu.get_cpu_energy().unwrap(), 100 + max);

        // Continues increasing after wraparound.
        write_energy(&path, 5_000);
        assert_eq!(cpu.get_cpu_energy().unwrap(), 5_000 + max);
    }

    #[test]
    fn multiple_wraparounds() {
        let tmp = tempfile::tempdir().unwrap();
        let max = 1_000;
        let (mut cpu, path, _) = make_test_cpu(tmp.path(), max, false);

        write_energy(&path, 800);
        assert_eq!(cpu.get_cpu_energy().unwrap(), 800);

        // First wraparound.
        write_energy(&path, 200);
        assert_eq!(cpu.get_cpu_energy().unwrap(), 200 + max);

        write_energy(&path, 900);
        assert_eq!(cpu.get_cpu_energy().unwrap(), 900 + max);

        // Second wraparound.
        write_energy(&path, 50);
        assert_eq!(cpu.get_cpu_energy().unwrap(), 50 + 2 * max);

        // Third wraparound.
        write_energy(&path, 30);
        assert_eq!(cpu.get_cpu_energy().unwrap(), 30 + 3 * max);
    }

    #[test]
    fn wraparound_to_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let max = 1_000_000;
        let (mut cpu, path, _) = make_test_cpu(tmp.path(), max, false);

        write_energy(&path, 500_000);
        assert_eq!(cpu.get_cpu_energy().unwrap(), 500_000);

        // Wraps to exactly 0.
        write_energy(&path, 0);
        assert_eq!(cpu.get_cpu_energy().unwrap(), max);
    }

    #[test]
    fn first_call_establishes_baseline() {
        let tmp = tempfile::tempdir().unwrap();
        let max = 1_000_000;
        let (mut cpu, path, _) = make_test_cpu(tmp.path(), max, false);

        // First call with a non-zero starting value (no wraparound offset).
        write_energy(&path, 42_000);
        assert_eq!(cpu.get_cpu_energy().unwrap(), 42_000);

        // Second call still higher (still no offset).
        write_energy(&path, 100_000);
        assert_eq!(cpu.get_cpu_energy().unwrap(), 100_000);
    }

    #[test]
    fn dram_wraparound() {
        let tmp = tempfile::tempdir().unwrap();
        let max = 500_000;
        let (mut cpu, _, dram_path) = make_test_cpu(tmp.path(), max, true);
        let dram_path = dram_path.unwrap();

        write_energy(&dram_path, 400_000);
        assert_eq!(cpu.get_dram_energy().unwrap(), 400_000);

        // DRAM wraps around.
        write_energy(&dram_path, 1_000);
        assert_eq!(cpu.get_dram_energy().unwrap(), 1_000 + max);

        // Continues after wraparound.
        write_energy(&dram_path, 200_000);
        assert_eq!(cpu.get_dram_energy().unwrap(), 200_000 + max);
    }

    #[test]
    fn cpu_and_dram_wraparound_independently() {
        let tmp = tempfile::tempdir().unwrap();
        let max = 1_000;
        let (mut cpu, cpu_path, dram_path) = make_test_cpu(tmp.path(), max, true);
        let dram_path = dram_path.unwrap();

        // Both start high.
        write_energy(&cpu_path, 800);
        write_energy(&dram_path, 600);
        assert_eq!(cpu.get_cpu_energy().unwrap(), 800);
        assert_eq!(cpu.get_dram_energy().unwrap(), 600);

        // Only CPU wraps.
        write_energy(&cpu_path, 100);
        write_energy(&dram_path, 900);
        assert_eq!(cpu.get_cpu_energy().unwrap(), 100 + max);
        assert_eq!(cpu.get_dram_energy().unwrap(), 900);

        // Only DRAM wraps.
        write_energy(&cpu_path, 500);
        write_energy(&dram_path, 200);
        assert_eq!(cpu.get_cpu_energy().unwrap(), 500 + max);
        assert_eq!(cpu.get_dram_energy().unwrap(), 200 + max);

        // Both wrap.
        write_energy(&cpu_path, 50);
        write_energy(&dram_path, 50);
        assert_eq!(cpu.get_cpu_energy().unwrap(), 50 + 2 * max);
        assert_eq!(cpu.get_dram_energy().unwrap(), 50 + 2 * max);
    }

    #[test]
    fn dram_not_available_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let (mut cpu, _, _) = make_test_cpu(tmp.path(), 1_000_000, false);

        assert!(cpu.get_dram_energy().is_err());
    }

    #[test]
    fn compensated_values_are_monotonic_under_rapid_wraparounds() {
        let tmp = tempfile::tempdir().unwrap();
        let max = 100;
        let (mut cpu, path, _) = make_test_cpu(tmp.path(), max, false);

        // Simulate many rapid wraparounds with a small max_energy_uj.
        // The raw counter cycles: 0 -> 80 -> 30 -> 90 -> 10 -> 70 -> ...
        let raw_sequence = [80, 30, 90, 10, 70, 20, 60, 5, 95, 0];
        write_energy(&path, raw_sequence[0]);
        let mut last_compensated = cpu.get_cpu_energy().unwrap();

        for &raw in &raw_sequence[1..] {
            write_energy(&path, raw);
            let compensated = cpu.get_cpu_energy().unwrap();
            assert!(
                compensated >= last_compensated,
                "Compensated energy decreased: {last_compensated} -> {compensated} (raw={raw})",
            );
            last_compensated = compensated;
        }
    }

    /// Write the power limit files of a RAPL zone. Each constraint is
    /// `(name, power_limit_uw, max_power_uw, time_window_us)`.
    fn write_zone_limits(dir: &Path, enabled: &str, constraints: &[(&str, u64, u64, u64)]) {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join("enabled"), format!("{enabled}\n")).unwrap();
        for (index, (name, power_limit_uw, max_power_uw, time_window_us)) in
            constraints.iter().enumerate()
        {
            let file = |field: &str| dir.join(format!("constraint_{index}_{field}"));
            fs::write(file("name"), format!("{name}\n")).unwrap();
            fs::write(file("power_limit_uw"), format!("{power_limit_uw}\n")).unwrap();
            fs::write(file("max_power_uw"), format!("{max_power_uw}\n")).unwrap();
            fs::write(file("time_window_us"), format!("{time_window_us}\n")).unwrap();
        }
    }

    /// Create a RaplCpu whose package and DRAM zones live in the given directories.
    fn make_limits_cpu(cpu_dir: &Path, dram_dir: Option<&Path>) -> RaplCpu {
        let zone = |dir: &Path, name: &str| {
            Arc::new(PackageInfo {
                index: 0,
                name: name.to_string(),
                zone_dir: dir.to_path_buf(),
                energy_uj_path: dir.join("energy_uj"),
                max_energy_uj: 1_000_000,
            })
        };
        RaplCpu {
            cpu: zone(cpu_dir, "package-0"),
            dram: dram_dir.map(|dir| zone(dir, "dram")),
            last_cpu_raw_uj: None,
            cpu_wraparound_count: 0,
            last_dram_raw_uj: None,
            dram_wraparound_count: 0,
            hsmp: None,
            original_power_limits: None,
        }
    }

    #[test]
    fn read_zone_limits_reads_all_constraints() {
        let tmp = tempfile::tempdir().unwrap();
        write_zone_limits(
            tmp.path(),
            "1",
            &[
                ("long_term", 205_000_000, 205_000_000, 999_424),
                ("short_term", 246_000_000, 780_000_000, 999_424),
            ],
        );

        assert_eq!(
            read_zone_limits(tmp.path()).unwrap(),
            ZonePowerLimits {
                enabled: true,
                constraints: vec![
                    PowerLimitConstraint {
                        name: "long_term".to_string(),
                        power_limit_mw: 205_000,
                        max_power_mw: Some(205_000),
                        time_window_us: Some(999_424),
                    },
                    PowerLimitConstraint {
                        name: "short_term".to_string(),
                        power_limit_mw: 246_000,
                        max_power_mw: Some(780_000),
                        time_window_us: Some(999_424),
                    },
                ],
            }
        );
    }

    #[test]
    fn read_zone_limits_without_constraints() {
        let tmp = tempfile::tempdir().unwrap();
        write_zone_limits(tmp.path(), "0", &[]);

        assert_eq!(
            read_zone_limits(tmp.path()).unwrap(),
            ZonePowerLimits {
                enabled: false,
                constraints: vec![],
            }
        );
    }

    #[test]
    fn read_zone_limits_missing_constraint_file_errors() {
        let tmp = tempfile::tempdir().unwrap();
        write_zone_limits(
            tmp.path(),
            "1",
            &[("long_term", 205_000_000, 205_000_000, 999_424)],
        );
        fs::remove_file(tmp.path().join("constraint_0_max_power_uw")).unwrap();

        assert!(read_zone_limits(tmp.path()).is_err());
    }

    /// A file reader that fails with the given errno for one file and reads
    /// every other file from disk.
    #[cfg(unix)]
    fn failing_reader(
        failing_file: &str,
        errno: nix::errno::Errno,
    ) -> impl Fn(&Path) -> std::io::Result<String> + '_ {
        move |path: &Path| {
            if path.ends_with(failing_file) {
                Err(std::io::Error::from_raw_os_error(errno as i32))
            } else {
                fs::read_to_string(path)
            }
        }
    }

    /// Kernels 6.5 and later answer `ENODATA` for the time window of `peak_power`.
    #[cfg(unix)]
    #[test]
    fn read_zone_limits_enodata_attribute_is_none() {
        let tmp = tempfile::tempdir().unwrap();
        write_zone_limits(
            tmp.path(),
            "1",
            &[
                ("long_term", 28_000_000, 28_000_000, 27_983_872),
                ("short_term", 64_000_000, 64_000_000, 2_440),
                ("peak_power", 121_000_000, 128_000_000, 0),
            ],
        );

        let limits = read_zone_limits_with(
            tmp.path(),
            failing_reader("constraint_2_time_window_us", nix::errno::Errno::ENODATA),
        )
        .unwrap();
        assert_eq!(limits.constraints.len(), 3);
        assert_eq!(
            limits.constraints[2],
            PowerLimitConstraint {
                name: "peak_power".to_string(),
                power_limit_mw: 121_000,
                max_power_mw: Some(128_000),
                time_window_us: None,
            }
        );
        assert_eq!(limits.constraints[0].time_window_us, Some(27_983_872));

        let limits = read_zone_limits_with(
            tmp.path(),
            failing_reader("constraint_0_max_power_uw", nix::errno::Errno::ENODATA),
        )
        .unwrap();
        assert_eq!(limits.constraints[0].max_power_mw, None);
    }

    #[cfg(unix)]
    #[test]
    fn read_zone_limits_other_read_errors_propagate() {
        let tmp = tempfile::tempdir().unwrap();
        write_zone_limits(
            tmp.path(),
            "1",
            &[("long_term", 205_000_000, 205_000_000, 999_424)],
        );

        for file in [
            "enabled",
            "constraint_0_name",
            "constraint_0_power_limit_uw",
            "constraint_0_max_power_uw",
            "constraint_0_time_window_us",
        ] {
            for errno in [nix::errno::Errno::EACCES, nix::errno::Errno::EIO] {
                assert!(
                    read_zone_limits_with(tmp.path(), failing_reader(file, errno)).is_err(),
                    "{file} failing with {errno} should be an error"
                );
            }
        }
        assert!(
            read_zone_limits_with(
                tmp.path(),
                failing_reader("constraint_0_power_limit_uw", nix::errno::Errno::ENODATA)
            )
            .is_err(),
            "power_limit_uw is mandatory in powercap, so ENODATA is an error"
        );
    }

    /// A constraint name Zeusd cannot read fails the read with the path
    /// instead of ending the constraint list there.
    #[cfg(unix)]
    #[test]
    fn unreadable_constraint_name_is_a_permission_error() {
        use actix_web::ResponseError;

        let tmp = tempfile::tempdir().unwrap();
        write_zone_limits(
            tmp.path(),
            "1",
            &[
                ("long_term", 205_000_000, 205_000_000, 999_424),
                ("short_term", 246_000_000, 780_000_000, 2_440),
            ],
        );

        for file in [
            "enabled",
            "constraint_1_name",
            "constraint_1_power_limit_uw",
        ] {
            let error =
                read_zone_limits_with(tmp.path(), failing_reader(file, nix::errno::Errno::EACCES))
                    .unwrap_err();
            assert!(
                matches!(&error, ZeusdError::CpuLimitReadError { path, .. } if path.ends_with(file)),
                "{error}"
            );
            assert_eq!(error.status_code(), actix_web::http::StatusCode::FORBIDDEN);
        }
    }

    #[test]
    fn power_limits_with_dram() {
        let tmp = tempfile::tempdir().unwrap();
        let cpu_dir = tmp.path().join("package");
        let dram_dir = cpu_dir.join("dram");
        write_zone_limits(
            &cpu_dir,
            "1",
            &[("long_term", 205_000_000, 205_000_000, 999_424)],
        );
        write_zone_limits(&dram_dir, "0", &[("long_term", 0, 121_000_000, 976)]);
        let cpu = make_limits_cpu(&cpu_dir, Some(&dram_dir));

        let limits = cpu.get_power_limits().unwrap();
        assert_eq!(limits.cpu, read_zone_limits(&cpu_dir).unwrap());
        assert_eq!(limits.dram, Some(read_zone_limits(&dram_dir).unwrap()));
    }

    #[test]
    fn power_limits_without_dram_serializes_null() {
        let tmp = tempfile::tempdir().unwrap();
        write_zone_limits(
            tmp.path(),
            "1",
            &[("long_term", 205_000_000, 205_000_000, 999_424)],
        );
        let cpu = make_limits_cpu(tmp.path(), None);

        assert_eq!(
            serde_json::to_value(cpu.get_power_limits().unwrap()).unwrap(),
            serde_json::json!({
                "cpu": {
                    "enabled": true,
                    "constraints": [{
                        "name": "long_term",
                        "power_limit_mw": 205000,
                        "max_power_mw": 205000,
                        "time_window_us": 999424,
                    }],
                },
                "dram": null,
            })
        );
    }

    fn read_file(dir: &Path, file: &str) -> String {
        fs::read_to_string(dir.join(file))
            .unwrap()
            .trim()
            .to_string()
    }

    fn intel_package(dir: &Path) -> RaplCpu {
        write_zone_limits(
            dir,
            "1",
            &[
                ("long_term", 205_000_000, 205_000_000, 999_424),
                ("short_term", 246_000_000, 780_000_000, 999_424),
            ],
        );
        make_limits_cpu(dir, None)
    }

    fn amd_package(dir: &Path, fake: Arc<FakeHsmp>) -> RaplCpu {
        write_zone_limits(dir, "0", &[]);
        let mut cpu = make_limits_cpu(dir, None);
        cpu.attach_hsmp(fake).unwrap();
        cpu
    }

    /// Record the current settings as the original settings, as startup does.
    fn record_original(cpu: &mut RaplCpu) {
        let (settings, hsmp_error) = cpu.power_limit_settings().unwrap();
        assert!(hsmp_error.is_none(), "{hsmp_error:?}");
        cpu.set_original_power_limits(settings.constraints);
    }

    #[test]
    fn set_writes_powercap_files_in_kernel_units() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cpu = intel_package(tmp.path());
        record_original(&mut cpu);

        cpu.set_power_limit("short_term", 150_000).unwrap();
        assert!(cpu
            .set_power_limit_time_window("socket", 27_983_872)
            .is_err());

        assert_eq!(
            read_file(tmp.path(), "constraint_1_power_limit_uw"),
            "150000000"
        );
        assert_eq!(
            read_file(tmp.path(), "constraint_0_power_limit_uw"),
            "205000000"
        );
        assert_eq!(
            read_file(tmp.path(), "constraint_0_time_window_us"),
            "999424"
        );
        assert!(cpu.set_power_limit("socket", 150_000).is_err());
        assert!(cpu.set_power_limit_time_window("socket", 1_000).is_err());
    }

    #[test]
    fn hsmp_socket_is_a_package_constraint() {
        let tmp = tempfile::tempdir().unwrap();
        let fake = FakeHsmp::new(vec![200_000], 280_000);
        let mut cpu = amd_package(tmp.path(), fake.clone());
        record_original(&mut cpu);

        assert_eq!(
            cpu.get_power_limits().unwrap().cpu,
            ZonePowerLimits {
                enabled: false,
                constraints: vec![PowerLimitConstraint {
                    name: "socket".to_string(),
                    power_limit_mw: 200_000,
                    max_power_mw: Some(280_000),
                    time_window_us: None,
                }],
            }
        );

        cpu.set_power_limit("socket", 150_000).unwrap();
        assert_eq!(*fake.limits_mw.lock().unwrap(), vec![150_000]);
        assert!(cpu.set_power_limit_time_window("socket", 1_000).is_err());
    }

    #[test]
    fn attach_hsmp_uses_physical_package_id() {
        let tmp = tempfile::tempdir().unwrap();
        let fake = FakeHsmp::new(vec![200_000, 190_000], 280_000);
        write_zone_limits(tmp.path(), "0", &[]);

        let mut cpu = make_limits_cpu(tmp.path(), None);
        cpu.cpu = Arc::new(PackageInfo {
            index: 0,
            name: "package-1".to_string(),
            zone_dir: tmp.path().to_path_buf(),
            energy_uj_path: tmp.path().join("energy_uj"),
            max_energy_uj: 1_000_000,
        });
        cpu.attach_hsmp(fake.clone()).unwrap();
        assert_eq!(
            cpu.get_power_limits().unwrap().cpu.constraints[0].power_limit_mw,
            190_000
        );

        cpu.cpu = Arc::new(PackageInfo {
            index: 0,
            name: "package-0-die-1".to_string(),
            zone_dir: tmp.path().to_path_buf(),
            energy_uj_path: tmp.path().join("energy_uj"),
            max_energy_uj: 1_000_000,
        });
        assert!(cpu.attach_hsmp(fake).is_err());
    }

    #[test]
    fn reset_restores_original() {
        let tmp = tempfile::tempdir().unwrap();
        let fake = FakeHsmp::new(vec![200_000], 280_000);
        let mut cpu = intel_package(tmp.path());
        cpu.attach_hsmp(fake.clone()).unwrap();
        let (original, hsmp_error) = cpu.power_limit_settings().unwrap();
        assert!(hsmp_error.is_none());
        assert_eq!(original.zone, "package-0");
        assert_eq!(
            original.constraints,
            vec![
                ConstraintSetting {
                    name: "long_term".to_string(),
                    power_limit_uw: 205_000_000,
                    time_window_us: Some(999_424),
                },
                ConstraintSetting {
                    name: "short_term".to_string(),
                    power_limit_uw: 246_000_000,
                    time_window_us: Some(999_424),
                },
                ConstraintSetting {
                    name: "socket".to_string(),
                    power_limit_uw: 200_000_000,
                    time_window_us: None,
                },
            ]
        );
        cpu.set_original_power_limits(original.constraints);

        cpu.set_power_limit("long_term", 100_000).unwrap();

        cpu.set_power_limit("socket", 150_000).unwrap();
        cpu.reset_power_limits().unwrap();

        assert_eq!(
            read_file(tmp.path(), "constraint_0_power_limit_uw"),
            "205000000"
        );
        assert_eq!(
            read_file(tmp.path(), "constraint_1_time_window_us"),
            "999424"
        );
        assert_eq!(*fake.limits_mw.lock().unwrap(), vec![200_000]);
    }

    /// A constraint locked by the BIOS cannot be written but also cannot
    /// differ from its original setting, so reset leaves it alone.
    #[cfg(unix)]
    #[test]
    fn reset_skips_settings_that_match_the_original() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let mut cpu = intel_package(tmp.path());
        record_original(&mut cpu);
        cpu.set_power_limit("short_term", 100_000).unwrap();

        for file in ["constraint_0_power_limit_uw", "constraint_0_time_window_us"] {
            fs::set_permissions(tmp.path().join(file), fs::Permissions::from_mode(0o444)).unwrap();
        }
        cpu.reset_power_limits().unwrap();
        assert_eq!(
            read_file(tmp.path(), "constraint_1_power_limit_uw"),
            "246000000"
        );
    }

    #[test]
    fn reset_without_original_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cpu = intel_package(tmp.path());
        assert!(matches!(
            cpu.reset_power_limits(),
            Err(ZeusdError::CpuOriginalPowerLimitsMissingError(0))
        ));
        assert!(matches!(
            cpu.set_power_limit("long_term", 100_000),
            Err(ZeusdError::CpuOriginalPowerLimitsMissingError(0))
        ));
    }

    #[test]
    fn reset_restores_power_when_time_window_control_is_unavailable() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cpu = intel_package(tmp.path());
        Arc::get_mut(&mut cpu.cpu).unwrap().name = "psys".to_string();
        record_original(&mut cpu);
        cpu.set_power_limit("long_term", 100_000).unwrap();
        cpu.set_power_limit("short_term", 100_000).unwrap();
        fs::write(tmp.path().join("constraint_0_time_window_us"), "2440").unwrap();

        let error = cpu.reset_power_limits().unwrap_err();
        assert!(error
            .to_string()
            .contains("no package MSR time-window control"));
        assert_eq!(read_file(tmp.path(), "constraint_0_time_window_us"), "2440");
        assert_eq!(
            read_file(tmp.path(), "constraint_0_power_limit_uw"),
            "205000000"
        );
        assert_eq!(
            read_file(tmp.path(), "constraint_1_power_limit_uw"),
            "246000000"
        );
    }

    /// A failed write does not stop the remaining constraints from being restored.
    #[cfg(unix)]
    #[test]
    fn reset_continues_past_failed_writes() {
        use std::os::unix::fs::PermissionsExt;

        if nix::unistd::geteuid().is_root() {
            // Root ignores file permissions, so no write can be made to fail.
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let mut cpu = intel_package(tmp.path());
        record_original(&mut cpu);
        cpu.set_power_limit("long_term", 100_000).unwrap();
        cpu.set_power_limit("short_term", 100_000).unwrap();

        fs::set_permissions(
            tmp.path().join("constraint_0_power_limit_uw"),
            fs::Permissions::from_mode(0o444),
        )
        .unwrap();
        let error = cpu.reset_power_limits().unwrap_err();
        assert!(
            matches!(error, ZeusdError::CpuControlError { .. }),
            "{error}"
        );
        assert_eq!(
            read_file(tmp.path(), "constraint_1_power_limit_uw"),
            "246000000"
        );
    }

    #[test]
    fn stored_power_limit_allows_rounding_but_not_masking() {
        assert!(stored_as_requested(205_000_000, 205_000_000));
        // Rounded down to a 1/8 W power unit.
        assert!(stored_as_requested(205_100_000, 205_000_000));
        // Rounded down to a 1 W power unit, the coarsest the kernel supports.
        assert!(stored_as_requested(205_999_999, 205_000_000));
        // 5000 W with a 1/8 W unit and a 15-bit field: 40000 & 0x7fff = 7232 units.
        assert!(!stored_as_requested(5_000_000_000, 904_000_000));
        assert!(!stored_as_requested(205_000_000, 206_000_000));
    }

    #[test]
    fn socket_power_limit_above_maximum_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let fake = FakeHsmp::new(vec![200_000], 280_000);
        let mut cpu = amd_package(tmp.path(), fake.clone());
        record_original(&mut cpu);

        assert!(matches!(
            cpu.set_power_limit("socket", 280_001),
            Err(ZeusdError::InvalidRequest(_))
        ));
        assert_eq!(*fake.limits_mw.lock().unwrap(), vec![200_000]);
        cpu.set_power_limit("socket", 280_000).unwrap();
        assert_eq!(*fake.limits_mw.lock().unwrap(), vec![280_000]);
    }

    /// Power limit queries and control work without energy read access, and
    /// energy reads report the missing access.
    #[cfg(unix)]
    #[test]
    fn init_and_limits_do_not_need_energy_read_access() {
        use std::os::unix::fs::PermissionsExt;

        if nix::unistd::geteuid().is_root() {
            // Root ignores file permissions, so no read can be made to fail.
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        write_zone_limits(
            tmp.path(),
            "1",
            &[("long_term", 205_000_000, 205_000_000, 999_424)],
        );
        fs::write(tmp.path().join("name"), "package-0\n").unwrap();
        fs::write(tmp.path().join("max_energy_range_uj"), "262143328850\n").unwrap();
        let energy_path = tmp.path().join("energy_uj");
        fs::write(&energy_path, "1\n").unwrap();
        fs::set_permissions(&energy_path, fs::Permissions::from_mode(0o000)).unwrap();

        let mut cpu = make_limits_cpu(tmp.path(), None);
        cpu.cpu = Arc::new(PackageInfo::new(tmp.path(), 0).unwrap());
        assert_eq!(cpu.get_power_limits().unwrap().cpu.constraints.len(), 1);
        record_original(&mut cpu);
        cpu.set_power_limit("long_term", 150_000).unwrap();

        let mut access_errors = cpu.check_energy_access();
        assert_eq!(access_errors.len(), 1);
        for error in [access_errors.remove(0), cpu.get_cpu_energy().unwrap_err()] {
            assert!(
                matches!(error, ZeusdError::CpuEnergyReadError { cpu: 0, .. }),
                "{error}"
            );
        }
    }

    #[test]
    fn failed_hsmp_reads_are_hsmp_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let fake = FakeHsmp::new(vec![200_000], 280_000);
        write_zone_limits(tmp.path(), "0", &[]);
        let mut cpu = make_limits_cpu(tmp.path(), None);
        cpu.cpu = Arc::new(PackageInfo {
            index: 0,
            name: "package-1".to_string(),
            zone_dir: tmp.path().to_path_buf(),
            energy_uj_path: tmp.path().join("energy_uj"),
            max_energy_uj: 1_000_000,
        });
        // The fake firmware has no socket 1, so every message fails.
        cpu.attach_hsmp(fake).unwrap();

        for error in [
            cpu.get_power_limits().unwrap_err(),
            cpu.get_power_limit_constraints().unwrap_err(),
            cpu.get_package_constraint("socket").unwrap_err(),
            cpu.set_power_limit("socket", 150_000).unwrap_err(),
        ] {
            assert!(
                matches!(error, ZeusdError::CpuHsmpError { write: false, .. }),
                "{error}"
            );
        }
    }

    /// RAPL constraints are recorded and controlled without HSMP, and the
    /// failed HSMP read is returned instead of dropped.
    #[test]
    fn failed_hsmp_reads_do_not_block_rapl() {
        let tmp = tempfile::tempdir().unwrap();
        // The fake firmware has no socket 0, so every message fails.
        let fake = FakeHsmp::new(vec![], 280_000);
        let mut cpu = intel_package(tmp.path());
        cpu.attach_hsmp(fake.clone()).unwrap();

        let (settings, hsmp_error) = cpu.power_limit_settings().unwrap();
        assert!(
            matches!(
                hsmp_error,
                Some(ZeusdError::CpuHsmpError { write: false, .. })
            ),
            "{hsmp_error:?}"
        );
        let names: Vec<&str> = settings
            .constraints
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        assert_eq!(names, ["long_term", "short_term"]);
        cpu.set_original_power_limits(settings.constraints);
        fake.sent.lock().unwrap().clear();

        for command in [
            CpuCommand::SetPowerLimit {
                constraint: "short_term".to_string(),
                power_limit_mw: 150_000,
            },
            CpuCommand::ResetPowerLimits,
        ] {
            command
                .execute(&mut cpu, std::time::Instant::now())
                .unwrap();
        }
        assert_eq!(
            read_file(tmp.path(), "constraint_1_power_limit_uw"),
            "246000000"
        );
        assert!(fake.sent.lock().unwrap().is_empty());
    }

    #[test]
    fn reset_restores_rapl_when_socket_cannot_be_read() {
        let tmp = tempfile::tempdir().unwrap();
        let fake = FakeHsmp::new(vec![200_000], 280_000);
        let mut cpu = intel_package(tmp.path());
        cpu.attach_hsmp(fake.clone()).unwrap();
        record_original(&mut cpu);

        // The socket becomes unreadable after its original setting was recorded.
        fake.limits_mw.lock().unwrap().clear();
        cpu.set_power_limit("long_term", 100_000).unwrap();
        let error = cpu.reset_power_limits().unwrap_err();
        assert!(
            matches!(error, ZeusdError::CpuHsmpError { write: false, .. }),
            "{error}"
        );
        assert_eq!(
            read_file(tmp.path(), "constraint_0_power_limit_uw"),
            "205000000"
        );

        // The HSMP device was missing when this Zeusd started.
        cpu.hsmp = None;
        cpu.set_power_limit("long_term", 100_000).unwrap();
        let error = cpu.reset_power_limits().unwrap_err();
        assert!(
            matches!(
                error,
                ZeusdError::CpuHsmpError {
                    source: HsmpError::DeviceMissing(_),
                    ..
                }
            ),
            "{error}"
        );
        assert_eq!(
            read_file(tmp.path(), "constraint_0_power_limit_uw"),
            "205000000"
        );
    }

    #[test]
    fn constraint_without_original_setting_is_not_changed() {
        let tmp = tempfile::tempdir().unwrap();
        let fake = FakeHsmp::new(vec![], 280_000);
        let mut cpu = amd_package(tmp.path(), fake.clone());
        let (settings, hsmp_error) = cpu.power_limit_settings().unwrap();
        assert!(hsmp_error.is_some());
        cpu.set_original_power_limits(settings.constraints);

        // The socket becomes readable after the original settings were recorded.
        fake.limits_mw.lock().unwrap().push(200_000);
        let error = cpu.set_power_limit("socket", 150_000).unwrap_err();
        assert!(
            matches!(
                &error,
                ZeusdError::CpuConstraintOriginalMissingError { cpu: 0, constraint }
                    if constraint == "socket"
            ),
            "{error}"
        );
        cpu.reset_power_limits().unwrap();
        assert_eq!(*fake.limits_mw.lock().unwrap(), vec![200_000]);
    }

    #[test]
    fn rapl_power_limit_above_tdp_is_accepted() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cpu = intel_package(tmp.path());
        record_original(&mut cpu);

        cpu.set_power_limit("long_term", 250_000).unwrap();
        assert_eq!(
            read_file(tmp.path(), "constraint_0_power_limit_uw"),
            "250000000"
        );
    }
}

#[cfg(test)]
mod platform_domain_tests {
    use super::*;

    #[test]
    fn psys_has_no_package_msr_power_info() {
        let tmp = tempfile::tempdir().unwrap();
        for (name, value) in [
            ("name", "psys"),
            ("energy_uj", "1"),
            ("max_energy_range_uj", "1000000"),
            ("enabled", "1"),
            ("constraint_0_name", "long_term"),
            ("constraint_0_power_limit_uw", "205000000"),
            ("constraint_0_max_power_uw", "205000000"),
            ("constraint_0_time_window_us", "999424"),
        ] {
            fs::write(tmp.path().join(name), value).unwrap();
        }
        let cpu = RaplCpu {
            cpu: Arc::new(PackageInfo::new(tmp.path(), 1).unwrap()),
            dram: None,
            last_cpu_raw_uj: None,
            cpu_wraparound_count: 0,
            last_dram_raw_uj: None,
            dram_wraparound_count: 0,
            hsmp: None,
            original_power_limits: None,
        };
        assert!(cpu.get_power_limits().is_ok());
        assert_eq!(
            cpu.get_power_limit_constraints().unwrap(),
            CpuPowerLimitConstraints {
                rapl: None,
                hsmp: None
            }
        );
    }
}
