//! Implements `NvmlGpu`, a GPU manager for NVIDIA GPUs using the NVML library.
//!
//! Note that NVML is only available on Linux.

use nvml_wrapper::enums::device::GpuLockedClocksSetting;
use nvml_wrapper::{error::NvmlError, Device, Nvml};

use crate::devices::gpu::GpuManager;
use crate::error::ZeusdError;

pub struct NvmlGpu<'n> {
    _nvml: &'static Nvml,
    device: Device<'n>,
}

impl NvmlGpu<'static> {
    pub fn init(index: u32) -> Result<Self, ZeusdError> {
        // `Device` needs to hold a reference to `Nvml`, meaning that `Nvml` must outlive `Device`.
        // We can achieve this by leaking a `Box` containing `Nvml` and holding a reference to it.
        // `Nvml` will actually live until the server terminates inside the GPU management task.
        let _nvml = Box::leak(Box::new(Nvml::init()?));
        let device = _nvml.device_by_index(index)?;
        Ok(Self { _nvml, device })
    }
}

impl GpuManager for NvmlGpu<'static> {
    fn device_count() -> Result<u32, ZeusdError> {
        match Nvml::init() {
            Ok(nvml) => match nvml.device_count() {
                Ok(count) => Ok(count),
                Err(e) => Err(ZeusdError::NvmlError(e)),
            },
            // Specifically catch this error that is thrown when GPU is not available
            Err(NvmlError::LibloadingError(e)) => {
                tracing::error!("Error initializing NVML, {}", e);
                Ok(0)
            }
            Err(e) => Err(ZeusdError::NvmlError(e)),
        }
    }

    #[inline]
    fn set_persistence_mode(&mut self, enabled: bool) -> Result<(), ZeusdError> {
        Ok(self.device.set_persistent(enabled)?)
    }

    #[inline]
    fn set_power_management_limit(&mut self, power_limit_mw: u32) -> Result<(), ZeusdError> {
        Ok(self.device.set_power_management_limit(power_limit_mw)?)
    }

    #[inline]
    fn set_gpu_locked_clocks(
        &mut self,
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    ) -> Result<(), ZeusdError> {
        let setting = GpuLockedClocksSetting::Numeric {
            min_clock_mhz,
            max_clock_mhz,
        };
        Ok(self.device.set_gpu_locked_clocks(setting)?)
    }

    #[inline]
    fn reset_gpu_locked_clocks(&mut self) -> Result<(), ZeusdError> {
        Ok(self.device.reset_gpu_locked_clocks()?)
    }

    #[inline]
    fn set_mem_locked_clocks(
        &mut self,
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    ) -> Result<(), ZeusdError> {
        Ok(self
            .device
            .set_mem_locked_clocks(min_clock_mhz, max_clock_mhz)?)
    }

    #[inline]
    fn reset_mem_locked_clocks(&mut self) -> Result<(), ZeusdError> {
        Ok(self.device.reset_mem_locked_clocks()?)
    }
}
