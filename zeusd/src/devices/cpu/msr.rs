//! Intel RAPL package power information and time-window control through MSRs.
//!
//! The kernel's powercap interface exposes only part of `MSR_PKG_POWER_INFO`,
//! so zeusd reads it through the `msr` driver's `/dev/cpu/<n>/msr`.

use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const MSR_RAPL_POWER_UNIT: u64 = 0x606;
const MSR_PKG_POWER_INFO: u64 = 0x614;
const MSR_PKG_POWER_LIMIT: u64 = 0x610;

/// Largest raw value of the 15-bit power limit fields in `MSR_PKG_POWER_LIMIT`.
const POWER_LIMIT_FIELD_MAX: u64 = 0x7fff;

/// Package power information that Intel CPUs report in `MSR_PKG_POWER_INFO`.
///
/// Values are converted with the same integer units the kernel uses for the
/// powercap interface, so they are comparable with `get_power_limit`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RaplPowerInfo {
    /// Thermal design power, which the kernel also reports as the `long_term` `max_power_mw`.
    pub thermal_spec_power_mw: u64,
    /// Lowest power limit Intel documents as allowed for the package.
    pub min_power_mw: u64,
    /// Highest power limit Intel documents as allowed for the package, which the
    /// kernel also reports as the `short_term` `max_power_mw`.
    pub max_power_mw: u64,
    /// Longest time window Intel documents as allowed for the package.
    pub max_time_window_us: u64,
    /// Largest power limit the `MSR_PKG_POWER_LIMIT` register can hold.
    pub power_limit_register_max_mw: u64,
}

/// Decode `MSR_RAPL_POWER_UNIT` and `MSR_PKG_POWER_INFO`.
pub fn decode_power_info(power_unit_msr: u64, power_info_msr: u64) -> RaplPowerInfo {
    // Same integer units as `rapl_default_check_unit` in the kernel.
    let power_unit_uw = 1_000_000u64 >> (power_unit_msr & 0xf);
    let time_unit_us = 1_000_000u64 >> ((power_unit_msr >> 16) & 0xf);
    let power_mw = |raw: u64| raw * power_unit_uw / 1000;
    // Same encoding as `rapl_default_compute_time_window`: 2^Y * (1 + F/4) time units.
    let window = (power_info_msr >> 48) & 0x3f;
    let (y, f) = (window & 0x1f, (window & 0x60) >> 5);
    RaplPowerInfo {
        thermal_spec_power_mw: power_mw(power_info_msr & 0x7fff),
        min_power_mw: power_mw((power_info_msr >> 16) & 0x7fff),
        max_power_mw: power_mw((power_info_msr >> 32) & 0x7fff),
        max_time_window_us: (1u64 << y) * (4 + f) * time_unit_us / 4,
        power_limit_register_max_mw: power_mw(POWER_LIMIT_FIELD_MAX),
    }
}

/// Read the package power information of the CPU package with the given
/// physical package ID (and die ID on CPUs with per-die RAPL).
pub fn read_power_info(
    sysfs_cpu_dir: &Path,
    dev_cpu_dir: &Path,
    package_id: u32,
    die_id: Option<u32>,
) -> Result<RaplPowerInfo, MsrError> {
    let cpu = find_cpu(sysfs_cpu_dir, package_id, die_id)?;
    let path = dev_cpu_dir.join(cpu.to_string()).join("msr");
    Ok(decode_power_info(
        read_msr(&path, MSR_RAPL_POWER_UNIT)?,
        read_msr(&path, MSR_PKG_POWER_INFO)?,
    ))
}

/// Why an MSR operation failed, with the action that fixes it.
#[derive(thiserror::Error, Debug)]
pub enum MsrError {
    #[error(
        "{0} does not exist. Load the msr kernel module with `sudo modprobe msr` \
         (in a container, also pass the /dev/cpu devices)."
    )]
    DriverMissing(PathBuf),
    #[error("Writing {path} was denied: {source}. Check kernel lockdown (/sys/kernel/security/lockdown), msr.allow_writes (/sys/module/msr/parameters/allow_writes), and device permissions. MSR reads may still be available.")]
    WriteDenied { path: PathBuf, source: io::Error },
    #[error("The BIOS locked MSR_PKG_POWER_LIMIT; time windows cannot be changed.")]
    Locked,
    #[error("{0}")]
    InvalidWindow(String),
    #[error("MSR time-window control requires an Intel x86-64 CPU with exponential RAPL time windows; this CPU layout is unsupported.")]
    UnsupportedLayout,
    #[error("The hardware did not apply the time-window write for '{constraint}'.")]
    WriteIgnored { constraint: String },
    #[error("{error}; rollback also failed: {restore}")]
    RestoreFailed { error: String, restore: String },
    #[error("Opening {0} requires CAP_SYS_RAWIO and permission to access the device. In a container, pass the device and grant CAP_SYS_RAWIO.")]
    PermissionDenied(PathBuf),
    #[error("This CPU does not implement MSR {register:#x} ({path}).")]
    Unsupported { path: PathBuf, register: u64 },
    #[error("No online CPU belongs to package {package_id}{}.", die_id.map(|d| format!(" die {d}")).unwrap_or_default())]
    NoOnlineCpu {
        package_id: u32,
        die_id: Option<u32>,
    },
    #[error("Failed to access {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
}

fn open_msr(path: &Path, writable: bool) -> Result<File, MsrError> {
    OpenOptions::new()
        .read(true)
        .write(writable)
        .open(path)
        .map_err(|source| match source.kind() {
            io::ErrorKind::NotFound => MsrError::DriverMissing(path.to_path_buf()),
            io::ErrorKind::PermissionDenied => MsrError::PermissionDenied(path.to_path_buf()),
            _ => MsrError::Io {
                path: path.to_path_buf(),
                source,
            },
        })
}

fn read_msr_file(file: &File, path: &Path, register: u64) -> Result<u64, MsrError> {
    let mut buf = [0u8; 8];
    read_exact_at(file, &mut buf, register).map_err(|source| {
        if source.raw_os_error() == Some(5) {
            MsrError::Unsupported {
                path: path.to_path_buf(),
                register,
            }
        } else {
            MsrError::Io {
                path: path.to_path_buf(),
                source,
            }
        }
    })?;
    Ok(u64::from_le_bytes(buf))
}

fn read_msr(path: &Path, register: u64) -> Result<u64, MsrError> {
    read_msr_file(&open_msr(path, false)?, path, register)
}

/// Operations available independently of Intel MSR access.
pub const MSR_AVAILABILITY: &str = "Energy monitoring, current-limit queries, and power-limit \
    changes remain available without MSR access. Intel hardware-range queries require MSR read \
    access; time-window changes and exact restoration of changed time windows require MSR write access. \
    AMD HSMP control does not require MSR access.";

/// Package time windows in Intel's exponential RAPL encoding.
///
/// Writes change only the selected seven-bit time-window field. Power limits,
/// enable bits, clamp bits, and the other time window retain their current values.
pub struct TimeWindows {
    file: File,
    path: PathBuf,
    unit_us: u64,
}

impl TimeWindows {
    /// Open MSRs for the given package and reject unsupported time-window layouts.
    pub fn open(
        sysfs_cpu_dir: &Path,
        dev_cpu_dir: &Path,
        package_id: u32,
        die_id: Option<u32>,
        writable: bool,
    ) -> Result<Self, MsrError> {
        check_time_window_platform()?;
        let cpu = find_cpu(sysfs_cpu_dir, package_id, die_id)?;
        Self::from_path(dev_cpu_dir.join(cpu.to_string()).join("msr"), writable)
    }

    fn from_path(path: PathBuf, writable: bool) -> Result<Self, MsrError> {
        let file = open_msr(&path, writable)?;
        let units = read_msr_file(&file, &path, MSR_RAPL_POWER_UNIT)?;
        read_msr_file(&file, &path, MSR_PKG_POWER_LIMIT)?;
        Ok(Self {
            file,
            path,
            unit_us: 1_000_000 >> ((units >> 16) & 0xf),
        })
    }

    fn read_register(&self) -> Result<u64, MsrError> {
        read_msr_file(&self.file, &self.path, MSR_PKG_POWER_LIMIT)
    }

    /// Read the time window of `constraint` in microseconds.
    pub fn read(&self, constraint: &str) -> Result<u64, MsrError> {
        let shift = window_shift(constraint)?;
        Ok(decode_window(
            (self.read_register()? >> shift) & 0x7f,
            self.unit_us,
        ))
    }

    /// Set a window, rounding down to an encodable value (at least one time unit).
    /// With `exact`, reject a value that cannot be represented exactly.
    pub fn set(&self, constraint: &str, window_us: u64, exact: bool) -> Result<(), MsrError> {
        let shift = window_shift(constraint)?;
        let bits = encode_window(window_us, self.unit_us, exact)?;
        let old = self.read_register()?;
        let mask = 0x7f << shift;
        let new = (old & !mask) | (bits << shift);
        if new == old {
            return Ok(());
        }
        if old & (1 << 63) != 0 {
            return Err(MsrError::Locked);
        }
        self.write_register(new)?;
        let verification = self.read_register().and_then(|stored| {
            if stored & mask == new & mask {
                Ok(())
            } else {
                Err(MsrError::WriteIgnored {
                    constraint: constraint.to_string(),
                })
            }
        });
        if let Err(error) = verification {
            // Read-modify-write again so rollback does not overwrite unrelated
            // fields another controller may have changed since the first write.
            let rollback = self.read_register().and_then(|current| {
                self.write_register((current & !mask) | (old & mask))?;
                if self.read_register()? & mask != old & mask {
                    return Err(MsrError::WriteIgnored {
                        constraint: constraint.to_string(),
                    });
                }
                Ok(())
            });
            return match rollback {
                Ok(()) => Err(error),
                Err(rollback) => Err(MsrError::RestoreFailed {
                    error: error.to_string(),
                    restore: rollback.to_string(),
                }),
            };
        }
        Ok(())
    }

    fn write_register(&self, value: u64) -> Result<(), MsrError> {
        write_exact_at(&self.file, &value.to_le_bytes(), MSR_PKG_POWER_LIMIT).map_err(|source| {
            if source.kind() == io::ErrorKind::PermissionDenied {
                MsrError::WriteDenied {
                    path: self.path.clone(),
                    source,
                }
            } else {
                MsrError::Io {
                    path: self.path.clone(),
                    source,
                }
            }
        })
    }
}

fn window_shift(constraint: &str) -> Result<u32, MsrError> {
    match constraint {
        "long_term" => Ok(17),
        "short_term" => Ok(49),
        _ => Err(MsrError::InvalidWindow(format!(
            "Constraint '{constraint}' has no supported MSR time-window field"
        ))),
    }
}

fn decode_window(bits: u64, unit_us: u64) -> u64 {
    (1u64 << (bits & 0x1f)) * (4 + ((bits >> 5) & 3)) * unit_us / 4
}

fn encode_window(window_us: u64, unit_us: u64, exact: bool) -> Result<u64, MsrError> {
    // Older powercap kernels decode with signed `1 << Y`, which overflows at
    // Y=31. New requests must remain readable through sysfs without MSR access.
    // Exact restoration can still reproduce a baseline from a newer kernel.
    let max_us = decode_window(if exact { 0x7f } else { 0x7e }, unit_us);
    if window_us == 0 || window_us > max_us {
        return Err(MsrError::InvalidWindow(format!(
            "Time window must be positive and at most {max_us} us; received {window_us} us. New windows exclude encodings that older Linux kernels cannot read correctly."
        )));
    }
    let bits = (0..=0x7f)
        .filter(|&bits| decode_window(bits, unit_us) <= window_us)
        .max_by_key(|&bits| decode_window(bits, unit_us))
        .unwrap_or(0);
    let actual_us = decode_window(bits, unit_us);
    if exact && actual_us != window_us {
        return Err(MsrError::InvalidWindow(format!(
            "Cannot restore time window {window_us} us exactly with time unit {unit_us} us"
        )));
    }
    Ok(bits)
}

fn check_time_window_platform() -> Result<(), MsrError> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        use std::arch::x86_64::__cpuid;
        let (vendor, cpu) = (__cpuid(0), __cpuid(1));
        let intel = vendor.ebx == u32::from_le_bytes(*b"Genu")
            && vendor.edx == u32::from_le_bytes(*b"ineI")
            && vendor.ecx == u32::from_le_bytes(*b"ntel");
        let model = ((cpu.eax >> 4) & 0xf) | ((cpu.eax >> 12) & 0xf0);
        let family = (cpu.eax >> 8) & 0xf;
        // Silvermont and Airmont SoCs use linear windows and different units.
        if intel && !(family == 6 && matches!(model, 0x37 | 0x4a | 0x4c | 0x5a)) {
            return Ok(());
        }
    }
    Err(MsrError::UnsupportedLayout)
}

#[cfg(unix)]
fn write_exact_at(file: &File, buf: &[u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, offset)
}

#[cfg(not(unix))]
fn write_exact_at(_file: &File, _buf: &[u8], _offset: u64) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "MSR writes are only available on Linux",
    ))
}

#[cfg(unix)]
fn read_exact_at(file: &fs::File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(not(unix))]
fn read_exact_at(_file: &fs::File, _buf: &mut [u8], _offset: u64) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "MSRs are only readable on Linux",
    ))
}

/// Find the lowest-numbered online CPU in the given package (and die).
fn find_cpu(sysfs_cpu_dir: &Path, package_id: u32, die_id: Option<u32>) -> Result<u32, MsrError> {
    let io_error = |path: &Path, source| MsrError::Io {
        path: path.to_path_buf(),
        source,
    };
    let read_id = |path: PathBuf| -> Result<Option<u32>, MsrError> {
        match fs::read_to_string(&path) {
            Ok(contents) => contents
                .trim()
                .parse()
                .map(Some)
                .map_err(|e| io_error(&path, io::Error::new(io::ErrorKind::InvalidData, e))),
            // Offline CPUs have no topology directory.
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(io_error(&path, e)),
        }
    };

    let mut cpus: Vec<u32> = fs::read_dir(sysfs_cpu_dir)
        .map_err(|e| io_error(sysfs_cpu_dir, e))?
        .filter_map(|entry| {
            let name = entry.ok()?.file_name();
            name.to_str()?.strip_prefix("cpu")?.parse().ok()
        })
        .collect();
    cpus.sort_unstable();
    for cpu in cpus {
        let topology = sysfs_cpu_dir.join(format!("cpu{cpu}")).join("topology");
        if read_id(topology.join("physical_package_id"))? != Some(package_id) {
            continue;
        }
        if let Some(die_id) = die_id {
            if read_id(topology.join("die_id"))? != Some(die_id) {
                continue;
            }
        }
        return Ok(cpu);
    }
    Err(MsrError::NoOnlineCpu { package_id, die_id })
}

/// Parse a RAPL package zone name, `package-<N>` or `package-<N>-die-<M>`.
pub fn parse_package_zone_name(name: &str) -> Option<(u32, Option<u32>)> {
    let rest = name.strip_prefix("package-")?;
    match rest.split_once("-die-") {
        Some((package, die)) => Some((package.parse().ok()?, Some(die.parse().ok()?))),
        None => Some((rest.parse().ok()?, None)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Registers read on an Intel Xeon Gold 6330.
    const XEON_6330_POWER_UNIT: u64 = 0x0000_0000_000a_0e03;
    const XEON_6330_POWER_INFO: u64 = 0x000f_1860_0388_0668;

    #[test]
    fn decodes_xeon_gold_6330() {
        assert_eq!(
            decode_power_info(XEON_6330_POWER_UNIT, XEON_6330_POWER_INFO),
            RaplPowerInfo {
                thermal_spec_power_mw: 205_000,
                min_power_mw: 113_000,
                max_power_mw: 780_000,
                // 2^15 time units of 976 us, as the kernel computes it.
                max_time_window_us: 31_981_568,
                power_limit_register_max_mw: 4_095_875,
            }
        );
    }

    #[test]
    fn decodes_fractional_time_window() {
        // The 6-bit field leaves one bit for F. Y = 3, F = 1: 2^3 * 1.25 = 10 time units of 976 us.
        let info = decode_power_info(XEON_6330_POWER_UNIT, 0x23u64 << 48);
        assert_eq!(info.max_time_window_us, 9_760);
    }

    #[test]
    fn parses_package_zone_names() {
        assert_eq!(parse_package_zone_name("package-1"), Some((1, None)));
        assert_eq!(
            parse_package_zone_name("package-0-die-1"),
            Some((0, Some(1)))
        );
        assert_eq!(parse_package_zone_name("dram"), None);
        assert_eq!(parse_package_zone_name("package-x"), None);
    }

    #[cfg(unix)]
    fn write_topology(sysfs: &Path, cpu: u32, package_id: u32, die_id: u32) {
        let topology = sysfs.join(format!("cpu{cpu}")).join("topology");
        fs::create_dir_all(&topology).unwrap();
        fs::write(
            topology.join("physical_package_id"),
            format!("{package_id}\n"),
        )
        .unwrap();
        fs::write(topology.join("die_id"), format!("{die_id}\n")).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn reads_from_lowest_cpu_of_the_package() {
        let tmp = tempfile::tempdir().unwrap();
        let sysfs = tmp.path().join("sys");
        let dev = tmp.path().join("dev");
        for (cpu, package_id) in [(0, 0), (1, 1), (10, 1), (2, 0)] {
            write_topology(&sysfs, cpu, package_id, 0);
        }
        // An offline CPU has no topology directory.
        fs::create_dir_all(sysfs.join("cpu5")).unwrap();
        fs::create_dir_all(sysfs.join("cpufreq")).unwrap();

        let mut msr = vec![0u8; 0x620];
        msr[0x606..0x60e].copy_from_slice(&XEON_6330_POWER_UNIT.to_le_bytes());
        msr[0x614..0x61c].copy_from_slice(&XEON_6330_POWER_INFO.to_le_bytes());
        fs::create_dir_all(dev.join("1")).unwrap();
        fs::write(dev.join("1").join("msr"), &msr).unwrap();

        let info = read_power_info(&sysfs, &dev, 1, None).unwrap();
        assert_eq!(info.thermal_spec_power_mw, 205_000);

        assert!(matches!(
            read_power_info(&sysfs, &dev, 0, None),
            Err(MsrError::DriverMissing(_))
        ));
        assert!(matches!(
            read_power_info(&sysfs, &dev, 2, None),
            Err(MsrError::NoOnlineCpu {
                package_id: 2,
                die_id: None
            })
        ));
        assert!(matches!(
            read_power_info(&sysfs, &dev, 1, Some(3)),
            Err(MsrError::NoOnlineCpu { .. })
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires root and an Intel CPU with the msr kernel module loaded"]
    fn reads_power_info_from_device() {
        let info = read_power_info(
            Path::new("/sys/devices/system/cpu"),
            Path::new("/dev/cpu"),
            0,
            None,
        )
        .unwrap();
        assert!(info.thermal_spec_power_mw > 0);
        assert!(info.min_power_mw <= info.thermal_spec_power_mw);
    }
}

#[cfg(all(test, unix))]
mod time_window_tests {
    use super::*;

    fn registers(dir: &Path, value: u64) -> TimeWindows {
        let path = dir.join("msr");
        let mut bytes = vec![0; 0x618];
        bytes[0x606..0x60e].copy_from_slice(&(10u64 << 16).to_le_bytes());
        bytes[0x610..0x618].copy_from_slice(&value.to_le_bytes());
        fs::write(&path, bytes).unwrap();
        TimeWindows::from_path(path, true).unwrap()
    }

    #[test]
    fn every_window_encoding_can_be_restored_exactly() {
        for unit in [30, 976, 1_000_000] {
            for bits in 0..=0x7f {
                let window = decode_window(bits, unit);
                assert_eq!(encode_window(window, unit, true).unwrap(), bits);
            }
        }
        assert_eq!(encode_window(2440, 976, true).unwrap(), 0x21);
        assert_eq!(encode_window(500, 976, false).unwrap(), 0);
        assert!(encode_window(500, 976, true).is_err());
        assert_eq!(
            decode_window(encode_window(500_000, 976, false).unwrap(), 976),
            499_712
        );
    }

    #[test]
    fn rejects_unrepresentable_windows_before_writing() {
        let tmp = tempfile::tempdir().unwrap();
        let original = 0x0015_87b0_0015_8668;
        let windows = registers(tmp.path(), original);
        for value in [0, (1 << 31) * 976, (1 << 32) * 976, u64::MAX] {
            assert!(matches!(
                windows.set("long_term", value, false),
                Err(MsrError::InvalidWindow(_))
            ));
            assert_eq!(windows.read_register().unwrap(), original);
        }
        assert!(windows.set("peak_power", 2440, false).is_err());
        assert_eq!(windows.read_register().unwrap(), original);
    }

    #[test]
    fn new_windows_remain_readable_by_older_powercap_kernels() {
        for unit in [30, 976, 1_000_000] {
            let maximum = decode_window(0x7e, unit);
            assert_eq!(encode_window(maximum, unit, false).unwrap(), 0x7e);
            assert!(encode_window(maximum + 1, unit, false).is_err());
            for fraction in 0..4 {
                let bits = 31 | (fraction << 5);
                let window = decode_window(bits, unit);
                assert!(encode_window(window, unit, false).is_err());
                assert_eq!(encode_window(window, unit, true).unwrap(), bits);
            }
        }
    }

    #[test]
    fn fractional_reset_preserves_other_register_fields() {
        let tmp = tempfile::tempdir().unwrap();
        let original = 0x0015_87b0_0015_8668;
        let windows = registers(tmp.path(), original);
        for (name, shift) in [("long_term", 17), ("short_term", 49)] {
            windows.set(name, 2440, true).unwrap();
            let stored = windows.read_register().unwrap();
            let mask = 0x7f << shift;
            assert_eq!(stored & !mask, original & !mask);
            assert_eq!(windows.read(name).unwrap(), 2440);
            windows.set(name, 999424, true).unwrap();
            assert_eq!(windows.read_register().unwrap(), original);
        }
    }

    #[test]
    fn locked_register_remains_unchanged() {
        let tmp = tempfile::tempdir().unwrap();
        let original = 0x8015_87b0_0015_8668;
        let windows = registers(tmp.path(), original);
        assert!(matches!(
            windows.set("short_term", 2440, true),
            Err(MsrError::Locked)
        ));
        assert_eq!(windows.read_register().unwrap(), original);
    }

    #[test]
    fn matching_window_needs_no_write_even_when_locked() {
        let tmp = tempfile::tempdir().unwrap();
        let original = 0x8015_87b0_0015_8668;
        let writable = registers(tmp.path(), original);
        let readonly = TimeWindows::from_path(writable.path.clone(), false).unwrap();
        readonly.set("long_term", 999424, true).unwrap();
        assert_eq!(readonly.read_register().unwrap(), original);
    }

    #[test]
    fn missing_device_explains_how_to_load_driver() {
        let tmp = tempfile::tempdir().unwrap();
        let error = TimeWindows::from_path(tmp.path().join("absent"), true)
            .err()
            .unwrap();
        assert!(matches!(error, MsrError::DriverMissing(_)));
        assert!(error.to_string().contains("modprobe msr"));
    }
}
