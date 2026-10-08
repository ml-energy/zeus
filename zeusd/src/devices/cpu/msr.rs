//! Intel RAPL package power information read from model-specific registers.
//!
//! The kernel's powercap interface exposes only part of `MSR_PKG_POWER_INFO`,
//! so zeusd reads it through the `msr` driver's `/dev/cpu/<n>/msr`.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const MSR_RAPL_POWER_UNIT: u64 = 0x606;
const MSR_PKG_POWER_INFO: u64 = 0x614;

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

/// Why an MSR could not be read, with the action that fixes it.
#[derive(thiserror::Error, Debug)]
pub enum MsrError {
    #[error(
        "{0} does not exist. Load the msr kernel module with `modprobe msr` \
         (in a container, also pass the /dev/cpu devices)."
    )]
    DriverMissing(PathBuf),
    #[error("Reading {0} requires root with CAP_SYS_RAWIO.")]
    PermissionDenied(PathBuf),
    #[error("This CPU does not implement MSR {register:#x} ({path}).")]
    Unsupported { path: PathBuf, register: u64 },
    #[error("No online CPU belongs to package {package_id}{}.", die_id.map(|d| format!(" die {d}")).unwrap_or_default())]
    NoOnlineCpu {
        package_id: u32,
        die_id: Option<u32>,
    },
    #[error("Failed to read {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
}

fn read_msr(path: &Path, register: u64) -> Result<u64, MsrError> {
    let file = fs::File::open(path).map_err(|source| match source.kind() {
        io::ErrorKind::NotFound => MsrError::DriverMissing(path.to_path_buf()),
        io::ErrorKind::PermissionDenied => MsrError::PermissionDenied(path.to_path_buf()),
        _ => MsrError::Io {
            path: path.to_path_buf(),
            source,
        },
    })?;
    let mut buf = [0u8; 8];
    read_exact_at(&file, &mut buf, register).map_err(|source| {
        // The msr driver answers EIO when the CPU faults on an unknown register.
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
