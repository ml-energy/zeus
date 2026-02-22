//! CPU power measurement with RAPL. Only supported on Linux.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::string::String;
use std::sync::Arc;

use crate::devices::cpu::{CpuManager, PackageInfo};
use crate::error::ZeusdError;

// NOTE: To support Zeusd deployment in a docker container, this should support
//       sysfs mounts under places like `/zeus_sys`.
static RAPL_DIR: &str = "/sys/class/powercap/intel-rapl";

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
            energy_uj_path: cpu_energy_path,
            max_energy_uj: cpu_max_energy,
        })
    }
}

impl CpuManager for RaplCpu {
    fn device_count() -> Result<usize, ZeusdError> {
        let mut index_count = 0;
        let base_path = PathBuf::from(RAPL_DIR);

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
        let base_path = PathBuf::from(format!("{RAPL_DIR}/intel-rapl:{index}"));
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
            energy_uj_path: cpu_energy_path.clone(),
            max_energy_uj,
        });

        let (dram, dram_path) = if with_dram {
            let dram_energy_path = dir.join("dram_energy_uj");
            write_energy(&dram_energy_path, 0);
            let dram_info = Arc::new(PackageInfo {
                index: 0,
                name: "dram".to_string(),
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
        // The raw counter cycles: 0 → 80 → 30 → 90 → 10 → 70 → ...
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
}
