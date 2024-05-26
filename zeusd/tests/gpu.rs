use nvml_wrapper::enums::device::GpuLockedClocksSetting;
use zeusd::devices::gpu::GpuManager;

struct TestGpu {
    pub index: u32,
    pub enabled_persistent: Option<bool>,
    pub power_limit: Option<u32>,
    pub locked_clocks_setting: Option<GpuLockedClocksSetting>,
    pub call_count: usize,
}

impl GpuManager for TestGpu {
    fn init(index: u32) -> Result<Self, nvml_wrapper::error::NvmlError> {
        todo!()
    }

    fn device_count() -> Result<u32, nvml_wrapper::error::NvmlError> {
        Ok(4)
    }

    fn set_persistent(&mut self, enabled: bool) -> Result<(), nvml_wrapper::error::NvmlError> {
        todo!()
    }

    fn set_power_management_limit(
        &mut self,
        power_limit: u32,
    ) -> Result<(), nvml_wrapper::error::NvmlError> {
        todo!()
    }

    fn set_gpu_locked_clocks(
        &mut self,
        setting: nvml_wrapper::enums::device::GpuLockedClocksSetting,
    ) -> Result<(), nvml_wrapper::error::NvmlError> {
        todo!()
    }
}
