//! CPU power measurement with RAPL. Only supported on Linux.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::string::String;
use std::sync::Arc;

use once_cell::sync::Lazy;

use crate::devices::cpu::{
    CpuManager, PackageInfo, RaplConstraint, RaplPowerLimits, RaplZoneLimits,
};
use crate::error::ZeusdError;

static SYS_RAPL_DIR: &str = "/sys/class/powercap/intel-rapl";

// Docker masks `/sys/devices/virtual/powercap` by default, so containerized
// deployments bind-mount the host's RAPL directories under `/zeus_sys` instead.
// Same convention as the Zeus Python package.
static CONTAINER_RAPL_DIR: &str = "/zeus_sys/class/powercap/intel-rapl";

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
        })
    }
}

impl PackageInfo {
    pub fn new(base_path: &Path, index: usize) -> anyhow::Result<Self, ZeusdError> {
        let cpu_name_path = base_path.join("name");
        let cpu_energy_path = base_path.join("energy_uj");
        let cpu_max_energy_path = base_path.join("max_energy_range_uj");

        if !cpu_name_path.exists() || !cpu_max_energy_path.exists() || !cpu_energy_path.exists() {
            return Err(ZeusdError::CpuInitializationError(index));
        }

        let cpu_name = fs::read_to_string(&cpu_name_path)?.trim_end().to_string();
        read_u64(&cpu_energy_path)?;
        let cpu_max_energy = read_u64(&cpu_max_energy_path)?;
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
            Err(_) => {
                tracing::error!("RAPL not available");
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
            Err(_) => {
                return Err(ZeusdError::CpuInitializationError(index));
            }
        };

        Ok((Arc::new(cpu_info), None))
    }

    fn get_cpu_energy(&mut self) -> Result<u64, ZeusdError> {
        let raw = read_u64(&self.cpu.energy_uj_path)?;
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
                let raw = read_u64(&dram.energy_uj_path)?;
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

    fn get_power_limits(&self) -> Result<RaplPowerLimits, ZeusdError> {
        Ok(RaplPowerLimits {
            cpu: read_zone_limits(&self.cpu.zone_dir)?,
            dram: self
                .dram
                .as_ref()
                .map(|dram| read_zone_limits(&dram.zone_dir))
                .transpose()?,
        })
    }
}

/// Read the power limit state of a RAPL powercap zone.
///
/// The kernel numbers constraints from zero without gaps, so reading stops at
/// the first index without a `constraint_K_name` file.
fn read_zone_limits(zone_dir: &Path) -> Result<RaplZoneLimits, ZeusdError> {
    read_zone_limits_with(zone_dir, |path| fs::read_to_string(path))
}

/// [`read_zone_limits`] with an injectable file reader.
///
/// A constraint attribute the kernel has no value for fails with `ENODATA`
/// (e.g., the time window of `peak_power`) and is read as `None`.
fn read_zone_limits_with(
    zone_dir: &Path,
    read_file: impl Fn(&Path) -> std::io::Result<String>,
) -> Result<RaplZoneLimits, ZeusdError> {
    let parse_u64 = |path: &Path| -> std::io::Result<u64> {
        read_file(path)?
            .trim()
            .parse()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    };
    let parse_optional_u64 = |path: &Path| -> std::io::Result<Option<u64>> {
        match parse_u64(path) {
            Ok(value) => Ok(Some(value)),
            Err(e) if is_enodata(&e) => Ok(None),
            Err(e) => Err(e),
        }
    };

    let enabled = match parse_u64(&zone_dir.join("enabled"))? {
        0 => false,
        1 => true,
        value => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unexpected value {value} in {}/enabled", zone_dir.display()),
            )
            .into())
        }
    };

    let mut constraints = Vec::new();
    for index in 0.. {
        let file = |field: &str| zone_dir.join(format!("constraint_{index}_{field}"));
        let name_path = file("name");
        if !name_path.exists() {
            break;
        }
        constraints.push(RaplConstraint {
            name: read_file(&name_path)?.trim_end().to_string(),
            power_limit_mw: parse_u64(&file("power_limit_uw"))? / 1000,
            max_power_mw: parse_optional_u64(&file("max_power_uw"))?.map(|uw| uw / 1000),
            time_window_us: parse_optional_u64(&file("time_window_us"))?,
        });
    }

    Ok(RaplZoneLimits {
        enabled,
        constraints,
    })
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
            RaplZoneLimits {
                enabled: true,
                constraints: vec![
                    RaplConstraint {
                        name: "long_term".to_string(),
                        power_limit_mw: 205_000,
                        max_power_mw: Some(205_000),
                        time_window_us: Some(999_424),
                    },
                    RaplConstraint {
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
            RaplZoneLimits {
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
            RaplConstraint {
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
}
