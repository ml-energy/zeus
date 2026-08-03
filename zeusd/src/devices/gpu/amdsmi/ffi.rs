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
#[allow(dead_code)]
pub const AMDSMI_STATUS_NOT_FOUND: AmdsmiStatus = 31;
#[allow(dead_code)]
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

/// ABI 24 GPU power information with power fields in W and voltage fields in mV.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AmdsmiPowerInfo24 {
    pub current_socket_power: u32,
    pub average_socket_power: u32,
    pub gfx_voltage: u32,
    pub soc_voltage: u32,
    pub mem_voltage: u32,
    pub power_limit: u32,
    pub reserved: [u32; 11],
}

/// ABI 25 GPU power information with power fields in W and voltage fields in mV.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AmdsmiPowerInfo25 {
    pub socket_power: u64,
    pub current_socket_power: u32,
    pub average_socket_power: u32,
    pub gfx_voltage: u32,
    pub soc_voltage: u32,
    pub mem_voltage: u32,
    pub power_limit: u32,
    pub reserved: [u32; 2],
}

/// ABI 26 GPU power information with power fields in W and voltage fields in mV.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AmdsmiPowerInfo26 {
    pub socket_power: u64,
    pub current_socket_power: u32,
    pub average_socket_power: u32,
    pub gfx_voltage: u64,
    pub soc_voltage: u64,
    pub mem_voltage: u64,
    pub power_limit: u32,
    pub reserved: [u64; 18],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AmdsmiVersion24And25 {
    pub year: u32,
    pub major: u32,
    pub minor: u32,
    pub release: u32,
    pub build: *const c_char,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AmdsmiVersion26 {
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

pub type AmdsmiInitFn = unsafe extern "C" fn(u64) -> AmdsmiStatus;
pub type AmdsmiShutDownFn = unsafe extern "C" fn() -> AmdsmiStatus;
pub type AmdsmiGetSocketHandlesFn =
    unsafe extern "C" fn(*mut u32, *mut AmdsmiSocketHandle) -> AmdsmiStatus;
pub type AmdsmiGetProcessorHandlesFn =
    unsafe extern "C" fn(AmdsmiSocketHandle, *mut u32, *mut AmdsmiProcessorHandle) -> AmdsmiStatus;
pub type AmdsmiGetGpuAsicInfoFn =
    unsafe extern "C" fn(AmdsmiProcessorHandle, *mut AmdsmiAsicInfo) -> AmdsmiStatus;
pub type AmdsmiGetGpuDeviceBdfFn =
    unsafe extern "C" fn(AmdsmiProcessorHandle, *mut Bdf) -> AmdsmiStatus;
pub type AmdsmiGetPowerCapInfoFn =
    unsafe extern "C" fn(AmdsmiProcessorHandle, u32, *mut AmdsmiPowerCapInfo) -> AmdsmiStatus;
pub type AmdsmiSetPowerCapFn =
    unsafe extern "C" fn(AmdsmiProcessorHandle, u32, u64) -> AmdsmiStatus;
pub type AmdsmiGetPowerInfoFn =
    unsafe extern "C" fn(AmdsmiProcessorHandle, *mut c_void) -> AmdsmiStatus;
pub type AmdsmiGetEnergyCountFn =
    unsafe extern "C" fn(AmdsmiProcessorHandle, *mut u64, *mut f32, *mut u64) -> AmdsmiStatus;
pub type AmdsmiGetClockInfoFn =
    unsafe extern "C" fn(AmdsmiProcessorHandle, u32, *mut AmdsmiClkInfo) -> AmdsmiStatus;
pub type AmdsmiGetLibVersionFn = unsafe extern "C" fn(*mut c_void) -> AmdsmiStatus;
pub type AmdsmiStatusCodeToStringFn =
    unsafe extern "C" fn(AmdsmiStatus, *mut *const c_char) -> AmdsmiStatus;
pub type AmdsmiSetGpuClkLimitFn =
    unsafe extern "C" fn(AmdsmiProcessorHandle, u32, u32, u64) -> AmdsmiStatus;
pub type AmdsmiSetGpuPerfLevelFn = unsafe extern "C" fn(AmdsmiProcessorHandle, u32) -> AmdsmiStatus;

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

    #[test]
    fn abi_24_struct_layouts() {
        assert_eq!(size_of::<AmdsmiPowerInfo24>(), 68);
        assert_eq!(offset_of!(AmdsmiPowerInfo24, current_socket_power), 0);
        assert_eq!(offset_of!(AmdsmiPowerInfo24, average_socket_power), 4);
        assert_eq!(size_of::<AmdsmiVersion24And25>(), 24);
        assert_eq!(offset_of!(AmdsmiVersion24And25, major), 4);
    }

    #[test]
    fn abi_25_struct_layouts() {
        assert_eq!(size_of::<AmdsmiPowerInfo25>(), 40);
        assert_eq!(offset_of!(AmdsmiPowerInfo25, current_socket_power), 8);
        assert_eq!(offset_of!(AmdsmiPowerInfo25, average_socket_power), 12);
    }

    #[test]
    fn abi_26_struct_layouts() {
        assert_eq!(size_of::<AmdsmiPowerInfo26>(), 192);
        assert_eq!(offset_of!(AmdsmiPowerInfo26, current_socket_power), 8);
        assert_eq!(offset_of!(AmdsmiPowerInfo26, average_socket_power), 12);
        assert_eq!(size_of::<AmdsmiVersion26>(), 24);
        assert_eq!(offset_of!(AmdsmiVersion26, major), 0);
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
