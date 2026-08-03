//! Raw bindings to `libamd_smi.so`.

use std::ffi::{c_char, c_void};
use std::fmt;

pub type AmdsmiProcessorHandle = *mut c_void;
pub type AmdsmiSocketHandle = *mut c_void;
pub type AmdsmiStatus = u32;

pub const AMDSMI_STATUS_SUCCESS: AmdsmiStatus = 0;
pub const AMDSMI_STATUS_INVAL: AmdsmiStatus = 1;
pub const AMDSMI_STATUS_NOT_SUPPORTED: AmdsmiStatus = 2;
pub const AMDSMI_STATUS_NO_PERM: AmdsmiStatus = 10;
pub const AMDSMI_STATUS_NOT_FOUND: AmdsmiStatus = 31;
pub const AMDSMI_STATUS_NOT_INIT: AmdsmiStatus = 32;

pub const AMDSMI_INIT_AMD_GPUS: u64 = 2;

pub const AMDSMI_CLK_TYPE_GFX: u32 = 0;
pub const AMDSMI_CLK_TYPE_MEM: u32 = 4;

pub const AMDSMI_CLK_LIMIT_MIN: u32 = 0;
pub const AMDSMI_CLK_LIMIT_MAX: u32 = 1;

/// Unavailable power value originating from u16 GPU metrics fields, which the library pre-fills with 0xFFFF.
pub const AMDSMI_POWER_NA: u32 = 0xFFFF;

pub const AMDSMI_DEV_PERF_LEVEL_AUTO: u32 = 0;

/// GPU power-cap information in uW.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AmdsmiPowerCapInfo {
    pub power_cap: u64,
    pub default_power_cap: u64,
    pub dpm_cap: u64,
    pub min_power_cap: u64,
    pub max_power_cap: u64,
    pub reserved: [u64; 3],
}

/// GPU clock information in MHz.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AmdsmiClkInfo {
    pub clk: u32,
    pub min_clk: u32,
    pub max_clk: u32,
    pub clk_locked: u8,
    pub clk_deep_sleep: u8,
    pub reserved: [u32; 4],
}

/// GPU power information with power fields in W and voltage fields in mV.
#[cfg(amdsmi_abi = "24")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AmdsmiPowerInfo {
    pub current_socket_power: u32,
    pub average_socket_power: u32,
    pub gfx_voltage: u32,
    pub soc_voltage: u32,
    pub mem_voltage: u32,
    pub power_limit: u32,
    pub reserved: [u32; 11],
}

/// GPU power information with power fields in W and voltage fields in mV.
#[cfg(amdsmi_abi = "25")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AmdsmiPowerInfo {
    pub socket_power: u64,
    pub current_socket_power: u32,
    pub average_socket_power: u32,
    pub gfx_voltage: u32,
    pub soc_voltage: u32,
    pub mem_voltage: u32,
    pub power_limit: u32,
    pub reserved: [u32; 2],
}

/// GPU power information with power fields in W and voltage fields in mV.
#[cfg(amdsmi_abi = "26")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AmdsmiPowerInfo {
    pub socket_power: u64,
    pub current_socket_power: u32,
    pub average_socket_power: u32,
    pub gfx_voltage: u64,
    pub soc_voltage: u64,
    pub mem_voltage: u64,
    pub power_limit: u32,
    pub reserved: [u64; 18],
}

#[cfg(any(amdsmi_abi = "24", amdsmi_abi = "25"))]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AmdsmiVersion {
    pub year: u32,
    pub major: u32,
    pub minor: u32,
    pub release: u32,
    pub build: *const c_char,
}

#[cfg(amdsmi_abi = "26")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AmdsmiVersion {
    pub major: u32,
    pub minor: u32,
    pub release: u32,
    pub build: *const c_char,
}

/// ASIC information returned by AMD SMI.
///
/// The real C struct is about 448 bytes on ABI 24 and about 896 bytes on ABI 26, and only `market_name` at offset 0 is read; 1024 bytes safely over-allocates the out-buffer for every supported ABI.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AmdsmiAsicInfo {
    pub market_name: [c_char; 256],
    pub _rest: [u8; 768],
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct Bdf(pub u64);

impl Bdf {
    pub fn function(self) -> u8 {
        (self.0 & 0x7) as u8
    }

    pub fn device(self) -> u8 {
        ((self.0 >> 3) & 0x1f) as u8
    }

    pub fn bus(self) -> u8 {
        ((self.0 >> 8) & 0xff) as u8
    }

    pub fn domain(self) -> u64 {
        self.0 >> 16
    }
}

impl fmt::Display for Bdf {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.domain(),
            self.bus(),
            self.device(),
            self.function()
        )
    }
}

extern "C" {
    pub fn amdsmi_init(init_flags: u64) -> AmdsmiStatus;
    pub fn amdsmi_shut_down() -> AmdsmiStatus;
    pub fn amdsmi_get_socket_handles(
        socket_count: *mut u32,
        socket_handles: *mut AmdsmiSocketHandle,
    ) -> AmdsmiStatus;
    pub fn amdsmi_get_processor_handles(
        socket_handle: AmdsmiSocketHandle,
        processor_count: *mut u32,
        processor_handles: *mut AmdsmiProcessorHandle,
    ) -> AmdsmiStatus;
    pub fn amdsmi_get_gpu_asic_info(
        processor_handle: AmdsmiProcessorHandle,
        info: *mut AmdsmiAsicInfo,
    ) -> AmdsmiStatus;
    pub fn amdsmi_get_gpu_device_bdf(
        processor_handle: AmdsmiProcessorHandle,
        bdf: *mut Bdf,
    ) -> AmdsmiStatus;
    /// Writes power-cap information in uW.
    pub fn amdsmi_get_power_cap_info(
        processor_handle: AmdsmiProcessorHandle,
        sensor_ind: u32,
        info: *mut AmdsmiPowerCapInfo,
    ) -> AmdsmiStatus;
    /// Sets the power cap in uW.
    pub fn amdsmi_set_power_cap(
        processor_handle: AmdsmiProcessorHandle,
        sensor_ind: u32,
        cap: u64,
    ) -> AmdsmiStatus;
    /// Writes power information whose power fields are in W.
    pub fn amdsmi_get_power_info(
        processor_handle: AmdsmiProcessorHandle,
        info: *mut AmdsmiPowerInfo,
    ) -> AmdsmiStatus;
    /// Writes an energy count, its resolution in uJ per count, and a timestamp in ns.
    pub fn amdsmi_get_energy_count(
        processor_handle: AmdsmiProcessorHandle,
        energy_accumulator: *mut u64,
        counter_resolution: *mut f32,
        timestamp: *mut u64,
    ) -> AmdsmiStatus;
    /// Writes GPU clock information in MHz.
    pub fn amdsmi_get_clock_info(
        processor_handle: AmdsmiProcessorHandle,
        clk_type: u32,
        info: *mut AmdsmiClkInfo,
    ) -> AmdsmiStatus;
    pub fn amdsmi_get_lib_version(version: *mut AmdsmiVersion) -> AmdsmiStatus;
    pub fn amdsmi_status_code_to_string(
        status: AmdsmiStatus,
        status_string: *mut *const c_char,
    ) -> AmdsmiStatus;

    /// Sets one GPU clock limit in MHz. Exported by libamd_smi since 24.7 (ROCm 6.3),
    /// which is zeusd's build floor, and the only clock setter remaining in ABI 27.
    pub fn amdsmi_set_gpu_clk_limit(
        processor_handle: AmdsmiProcessorHandle,
        clk_type: u32,
        limit_type: u32,
        clk_value: u64,
    ) -> AmdsmiStatus;

    /// Sets the GPU performance level. Leaving manual mode restores default clock limits, while clock-limit writes force manual mode.
    pub fn amdsmi_set_gpu_perf_level(
        processor_handle: AmdsmiProcessorHandle,
        perf_level: u32,
    ) -> AmdsmiStatus;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    // Expected values come from amdsmi-abi/<version>/amdsmi_wrapper.py.
    #[test]
    fn common_struct_layouts() {
        assert_eq!(size_of::<AmdsmiPowerCapInfo>(), 64);
        assert_eq!(size_of::<AmdsmiClkInfo>(), 32);
        assert_eq!(size_of::<AmdsmiAsicInfo>(), 1024);
        assert_eq!(offset_of!(AmdsmiAsicInfo, market_name), 0);
        assert_eq!(size_of::<Bdf>(), 8);
    }

    #[cfg(amdsmi_abi = "24")]
    #[test]
    fn abi_24_struct_layouts() {
        assert_eq!(size_of::<AmdsmiPowerInfo>(), 68);
        assert_eq!(offset_of!(AmdsmiPowerInfo, current_socket_power), 0);
        assert_eq!(offset_of!(AmdsmiPowerInfo, average_socket_power), 4);
        assert_eq!(size_of::<AmdsmiVersion>(), 24);
        assert_eq!(offset_of!(AmdsmiVersion, major), 4);
    }

    #[cfg(amdsmi_abi = "25")]
    #[test]
    fn abi_25_struct_layouts() {
        assert_eq!(size_of::<AmdsmiPowerInfo>(), 40);
        assert_eq!(offset_of!(AmdsmiPowerInfo, current_socket_power), 8);
        assert_eq!(offset_of!(AmdsmiPowerInfo, average_socket_power), 12);
        assert_eq!(size_of::<AmdsmiVersion>(), 24);
        assert_eq!(offset_of!(AmdsmiVersion, major), 4);
    }

    #[cfg(amdsmi_abi = "26")]
    #[test]
    fn abi_26_struct_layouts() {
        assert_eq!(size_of::<AmdsmiPowerInfo>(), 192);
        assert_eq!(offset_of!(AmdsmiPowerInfo, current_socket_power), 8);
        assert_eq!(offset_of!(AmdsmiPowerInfo, average_socket_power), 12);
        assert_eq!(size_of::<AmdsmiVersion>(), 24);
        assert_eq!(offset_of!(AmdsmiVersion, major), 0);
    }

    #[test]
    fn bdf_fields_and_display() {
        let bdf = Bdf((1_u64 << 16) | (0xc5_u64 << 8));

        assert_eq!(bdf.domain(), 0x0001);
        assert_eq!(bdf.bus(), 0xc5);
        assert_eq!(bdf.device(), 0x00);
        assert_eq!(bdf.function(), 0);
        assert_eq!(bdf.to_string(), "0001:c5:00.0");
    }
}
