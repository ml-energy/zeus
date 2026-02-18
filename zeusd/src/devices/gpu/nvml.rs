//! Implements `NvmlGpu`, a GPU manager for NVIDIA GPUs using the NVML library.
//!
//! Note that NVML is only available on Linux.

use std::ffi::OsStr;

use nvml_wrapper::enums::device::GpuLockedClocksSetting;
use nvml_wrapper::{error::NvmlError, Device, Nvml};

use crate::devices::gpu::GpuManager;
use crate::error::ZeusdError;

pub struct NvmlGpu<'n> {
    _nvml: &'static Nvml,
    device: Device<'n>,
}

fn init_nvml() -> Result<Nvml, NvmlError> {
    // Initialize NVML and return the instance.
    match Nvml::init() {
        Err(NvmlError::LibloadingError(_)) => {
            tracing::warn!("NVML library not found, trying with `libnvidia-ml.so.1`");
            Nvml::builder()
                .lib_path(OsStr::new("libnvidia-ml.so.1"))
                .init()
        }
        res => res,
    }
}

impl NvmlGpu<'static> {
    pub fn init(index: u32) -> Result<Self, ZeusdError> {
        // `Device` needs to hold a reference to `Nvml`, meaning that `Nvml` must outlive `Device`.
        // We can achieve this by leaking a `Box` containing `Nvml` and holding a reference to it.
        // `Nvml` will actually live until the server terminates inside the GPU management task.
        let _nvml = Box::leak(Box::new(init_nvml()?));
        let device = _nvml.device_by_index(index)?;
        Ok(Self { _nvml, device })
    }
}

impl GpuManager for NvmlGpu<'static> {
    fn device_count() -> Result<u32, ZeusdError> {
        match init_nvml() {
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

    #[inline]
    fn get_total_energy_consumption(&mut self) -> Result<u64, ZeusdError> {
        Ok(self.device.total_energy_consumption()?)
    }

    fn get_instant_power_mw(&mut self) -> Result<u32, ZeusdError> {
        use nvml_wrapper::enums::device::SampleValue;
        use nvml_wrapper::structs::device::FieldId;
        use nvml_wrapper::sys_exports::field_id::NVML_FI_DEV_POWER_INSTANT;

        let results = self
            .device
            .field_values_for(&[FieldId(NVML_FI_DEV_POWER_INSTANT)])?;
        match results.into_iter().next() {
            Some(Ok(sample)) => match sample.value {
                Ok(SampleValue::U32(v)) => Ok(v),
                Ok(SampleValue::U64(v)) => Ok(v as u32),
                Ok(_) => Err(ZeusdError::NvmlError(NvmlError::InvalidArg)),
                Err(e) => Err(ZeusdError::NvmlError(e)),
            },
            Some(Err(e)) => Err(ZeusdError::NvmlError(e)),
            None => Err(ZeusdError::NvmlError(NvmlError::InvalidArg)),
        }
    }
}
