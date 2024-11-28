//! CPU power measurement with RAPL. Only supported on Linux.

use once_cell::sync::OnceCell;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::string::String;
use std::sync::{Arc, RwLock};
use tokio::io::AsyncReadExt;
use tokio::task::JoinHandle;
use tokio::time::{sleep, Duration};

use crate::devices::cpu::{CpuManager, PackageInfo};
use crate::error::ZeusdError;

// NOTE: To support Zeusd deployment in a docker container, this should support
//       sysfs mounts under places like `/zeus_sys`.
static RAPL_DIR: &str = "/sys/class/powercap/intel-rapl";

// Assuming a maximum power draw of 1000 Watts when we are polling every 0.1 seconds, the maximum
// amount the RAPL counter would increase (1000 * 1e6 * 0.1)
static RAPL_COUNTER_MAX_INCREASE: u64 = 1000 * 100000;

pub struct RaplCpu {
    cpu: Arc<PackageInfo>,
    dram: Option<Arc<PackageInfo>>,
    cpu_monitoring_task: OnceCell<JoinHandle<Result<(), ZeusdError>>>,
    dram_monitoring_task: OnceCell<JoinHandle<Result<(), ZeusdError>>>,
}

impl RaplCpu {
    pub fn init(_index: usize) -> Result<Self, ZeusdError> {
        let fields = RaplCpu::get_available_fields(_index)?;
        Ok(Self {
            cpu: fields.0,
            dram: fields.1,
            cpu_monitoring_task: OnceCell::new(),
            dram_monitoring_task: OnceCell::new(),
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
        // Try reding from energy_uj file
        read_u64(&cpu_energy_path)?;
        let cpu_max_energy = read_u64(&cpu_max_energy_path)?;
        let wraparound_counter = RwLock::new(0);
        Ok(PackageInfo {
            index,
            name: cpu_name,
            energy_uj_path: cpu_energy_path,
            max_energy_uj: cpu_max_energy,
            num_wraparounds: wraparound_counter,
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
        let base_path = PathBuf::from(format!("{}/intel-rapl:{}", RAPL_DIR, index));
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
        // Assume that RAPL counter will not wrap around twice during a request to poll energy. The
        // number of wrap arounds is polled twice to handle the case where the counter wraps around
        // a request. If this happens, `measurement` has to be updated as to not return an
        // unexpectedly large energy value.

        let handle = self
            .cpu_monitoring_task
            .get_or_init(|| tokio::spawn(monitor_rapl(Arc::clone(&self.cpu))));
        if handle.is_finished() {
            return Err(ZeusdError::CpuManagementTaskTerminatedError(self.cpu.index));
        }

        let num_wraparounds_before = *self
            .cpu
            .num_wraparounds
            .read()
            .map_err(|_| ZeusdError::CpuManagementTaskTerminatedError(self.cpu.index))?;
        let mut measurement = read_u64(&self.cpu.energy_uj_path)?;
        let num_wraparounds = *self
            .cpu
            .num_wraparounds
            .read()
            .map_err(|_| ZeusdError::CpuManagementTaskTerminatedError(self.cpu.index))?;
        if num_wraparounds != num_wraparounds_before {
            // Wraparound has happened after measurement, take measurement again
            measurement = read_u64(&self.cpu.energy_uj_path)?;
        }

        Ok(measurement + num_wraparounds * self.cpu.max_energy_uj)
    }

    fn get_dram_energy(&mut self) -> Result<u64, ZeusdError> {
        match &self.dram {
            None => Err(ZeusdError::CpuManagementTaskTerminatedError(self.cpu.index)),
            Some(dram) => {
                let handle = self
                    .dram_monitoring_task
                    .get_or_init(|| tokio::spawn(monitor_rapl(Arc::clone(dram))));
                if handle.is_finished() {
                    return Err(ZeusdError::CpuManagementTaskTerminatedError(dram.index));
                }

                let num_wraparounds_before = *dram
                    .num_wraparounds
                    .read()
                    .map_err(|_| ZeusdError::CpuManagementTaskTerminatedError(dram.index))?;
                let mut measurement = read_u64(&dram.energy_uj_path)?;
                let num_wraparounds = *dram
                    .num_wraparounds
                    .read()
                    .map_err(|_| ZeusdError::CpuManagementTaskTerminatedError(dram.index))?;
                if num_wraparounds != num_wraparounds_before {
                    // Wraparound has happened after measurement, take measurement again
                    measurement = read_u64(&dram.energy_uj_path)?;
                }

                Ok(measurement + num_wraparounds * dram.max_energy_uj)
            }
        }
    }

    fn stop_monitoring(&mut self) {
        if let Some(handle) = self.cpu_monitoring_task.take() {
            handle.abort();
        }
        if let Some(handle) = self.dram_monitoring_task.take() {
            handle.abort();
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

async fn read_u64_async(path: &PathBuf) -> Result<u64, std::io::Error> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut buf = String::new();
    file.read_to_string(&mut buf).await?;
    buf.trim()
        .parse()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

async fn monitor_rapl(rapl_file: Arc<PackageInfo>) -> Result<(), ZeusdError> {
    let mut last_energy_uj = read_u64_async(&rapl_file.energy_uj_path).await?;
    tracing::info!(
        "Monitoring started for {}",
        rapl_file.energy_uj_path.display()
    );
    loop {
        let current_energy_uj = read_u64_async(&rapl_file.energy_uj_path).await?;

        if current_energy_uj < last_energy_uj {
            let mut wraparound_guard = rapl_file
                .num_wraparounds
                .write()
                .map_err(|_| ZeusdError::CpuManagementTaskTerminatedError(rapl_file.index))?;
            *wraparound_guard += 1;
        }
        last_energy_uj = current_energy_uj;
        let sleep_time = if rapl_file.max_energy_uj - current_energy_uj < RAPL_COUNTER_MAX_INCREASE
        {
            100
        } else {
            1000
        };
        sleep(Duration::from_millis(sleep_time)).await;
    }
}
