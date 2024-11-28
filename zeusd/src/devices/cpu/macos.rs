//! Fake `RaplCpu` implementation to allow development and testing on MacOS.
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::devices::cpu::{CpuManager, PackageInfo};
use crate::error::ZeusdError;

pub struct RaplCpu {}

impl RaplCpu {
    pub fn init(_index: usize) -> Result<Self, ZeusdError> {
        Ok(Self {})
    }
}

impl CpuManager for RaplCpu {
    fn device_count() -> Result<usize, ZeusdError> {
        Ok(1)
    }

    fn get_available_fields(
        _index: usize,
    ) -> Result<(Arc<PackageInfo>, Option<Arc<PackageInfo>>), ZeusdError> {
        Ok((
            Arc::new(PackageInfo {
                index: _index,
                name: "package-0".to_string(),
                energy_uj_path: PathBuf::from(
                    "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
                ),
                max_energy_uj: 1000000,
                num_wraparounds: RwLock::new(0),
            }),
            Some(Arc::new(PackageInfo {
                index: _index,
                name: "dram".to_string(),
                energy_uj_path: PathBuf::from(
                    "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj",
                ),
                max_energy_uj: 1000000,
                num_wraparounds: RwLock::new(0),
            })),
        ))
    }

    fn get_cpu_energy(&mut self) -> Result<u64, ZeusdError> {
        Ok(10001)
    }

    fn get_dram_energy(&mut self) -> Result<u64, ZeusdError> {
        Ok(1001)
    }

    fn stop_monitoring(&mut self) {}

    fn is_dram_available(&self) -> bool {
        true
    }
}
