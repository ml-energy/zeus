//! Fake `NvmlGpu` implementation to allow development and testing on MacOS.

use crate::devices::gpu::GpuManager;
use crate::error::ZeusdError;

pub struct NvmlGpu;

impl NvmlGpu {
    pub fn init(_index: u32) -> Result<Self, ZeusdError> {
        Ok(Self)
    }
}

impl GpuManager for NvmlGpu {
    fn device_count() -> Result<u32, ZeusdError> {
        Ok(1)
    }

    fn set_persistence_mode(&mut self, _enabled: bool) -> Result<(), ZeusdError> {
        Ok(())
    }

    fn set_power_management_limit(&mut self, _power_limit_mw: u32) -> Result<(), ZeusdError> {
        Ok(())
    }

    fn set_gpu_locked_clocks(
        &mut self,
        _min_clock_mhz: u32,
        _max_clock_mhz: u32,
    ) -> Result<(), ZeusdError> {
        Ok(())
    }

    fn reset_gpu_locked_clocks(&mut self) -> Result<(), ZeusdError> {
        Ok(())
    }

    fn set_mem_locked_clocks(
        &mut self,
        _min_clock_mhz: u32,
        _max_clock_mhz: u32,
    ) -> Result<(), ZeusdError> {
        Ok(())
    }

    fn reset_mem_locked_clocks(&mut self) -> Result<(), ZeusdError> {
        Ok(())
    }
}
