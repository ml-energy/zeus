use zeusd::error::ZeusdError;
use zeusd::devices::gpu::GpuManager;

struct TestGpu {
    pub index: u32,
    pub enabled_persistent: Option<bool>,
    pub power_limit: Option<u32>,
    pub locked_clocks_setting: Option<(u32, u32)>,
    pub call_count: usize,
}

impl GpuManager for TestGpu {
    fn init(index: u32) -> Result<Self, ZeusdError> {
        todo!()
    }

    fn device_count() -> Result<u32, ZeusdError> {
        Ok(4)
    }

    fn set_persistent_mode(&mut self, enabled: bool) -> Result<(), ZeusdError> {
        todo!()
    }

    fn set_power_management_limit(&mut self, power_limit: u32) -> Result<(), ZeusdError> {
        todo!()
    }

    fn set_gpu_locked_clocks(&mut self, min_clock_mhz: u32, max_clock_mhz: u32) -> Result<(), ZeusdError> {
        todo!()
    }

    fn set_mem_locked_clocks(&mut self, min_clock_mhz: u32, max_clock_mhz: u32) -> Result<(), ZeusdError> {
        todo!()
    }
}
