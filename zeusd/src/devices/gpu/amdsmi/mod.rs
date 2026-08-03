//! Implements `AmdsmiGpu`, a GPU manager for AMD GPUs using AMD SMI.
//!
//! AMD SMI is available on Linux as part of ROCm. The library is initialized
//! once and remains initialized for the lifetime of the daemon.

use std::mem::MaybeUninit;
use std::ptr;

use once_cell::sync::OnceCell;

use crate::devices::gpu::GpuManager;
use crate::error::ZeusdError;

pub mod ffi;

static AMDSMI_INIT_STATUS: OnceCell<ffi::AmdsmiStatus> = OnceCell::new();

#[derive(Clone, Copy, Debug)]
enum PowerReading {
    Current,
    Average,
    Unsupported,
}

/// A GPU managed through AMD SMI.
pub struct AmdsmiGpu {
    handle: ffi::AmdsmiProcessorHandle,
    power_reading: PowerReading,
    // Single-task ownership serializes cache updates; failed writes requery bounds to cover external changes.
    gfx_clock_bounds: Option<(u32, u32)>,
    mem_clock_bounds: Option<(u32, u32)>,
}

// SAFETY: Processor handles are process-wide identifiers into the thread-safe
// AMD SMI library, not thread-local resources.
unsafe impl Send for AmdsmiGpu {}

fn amdsmi_error(status: ffi::AmdsmiStatus) -> ZeusdError {
    let mut message_ptr = ptr::null();
    // SAFETY: AMD SMI writes a borrowed pointer to a static status string.
    let message_status = unsafe { ffi::amdsmi_status_code_to_string(status, &mut message_ptr) };
    let msg = if message_status == ffi::AMDSMI_STATUS_SUCCESS && !message_ptr.is_null() {
        // SAFETY: A successful call returns a valid NUL-terminated string.
        unsafe { std::ffi::CStr::from_ptr(message_ptr) }
            .to_string_lossy()
            .into_owned()
    } else {
        status.to_string()
    };
    ZeusdError::AmdSmiError { status, msg }
}

fn check_status(status: ffi::AmdsmiStatus) -> Result<(), ZeusdError> {
    if status == ffi::AMDSMI_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(amdsmi_error(status))
    }
}

fn init_amdsmi() -> Result<(), ZeusdError> {
    let status = AMDSMI_INIT_STATUS.get_or_init(|| {
        // SAFETY: This process-global call is executed exactly once by `OnceCell`.
        unsafe { ffi::amdsmi_init(ffi::AMDSMI_INIT_AMD_GPUS) }
    });
    check_status(*status)
}

fn socket_handles() -> Result<Vec<ffi::AmdsmiSocketHandle>, ZeusdError> {
    let mut count = 0;
    // SAFETY: A null buffer requests the required socket count.
    check_status(unsafe { ffi::amdsmi_get_socket_handles(&mut count, ptr::null_mut()) })?;
    let mut handles = vec![ptr::null_mut(); count as usize];
    if count > 0 {
        // SAFETY: `handles` has room for the count returned by the first call.
        check_status(unsafe { ffi::amdsmi_get_socket_handles(&mut count, handles.as_mut_ptr()) })?;
        handles.truncate(count as usize);
    }
    Ok(handles)
}

fn processor_handles(
    socket: ffi::AmdsmiSocketHandle,
) -> Result<Vec<ffi::AmdsmiProcessorHandle>, ZeusdError> {
    let mut count = 0;
    // SAFETY: A null buffer requests the required processor count.
    check_status(unsafe {
        ffi::amdsmi_get_processor_handles(socket, &mut count, ptr::null_mut())
    })?;
    let mut handles = vec![ptr::null_mut(); count as usize];
    if count > 0 {
        // SAFETY: `handles` has room for the count returned by the first call.
        check_status(unsafe {
            ffi::amdsmi_get_processor_handles(socket, &mut count, handles.as_mut_ptr())
        })?;
        handles.truncate(count as usize);
    }
    Ok(handles)
}

fn enumerate_gpus() -> Result<Vec<ffi::AmdsmiProcessorHandle>, ZeusdError> {
    init_amdsmi()?;
    let mut handles = Vec::new();
    for socket in socket_handles()? {
        handles.extend(processor_handles(socket)?);
    }
    Ok(handles)
}

fn power_info(handle: ffi::AmdsmiProcessorHandle) -> Result<ffi::AmdsmiPowerInfo, ZeusdError> {
    let mut info = MaybeUninit::zeroed();
    // SAFETY: `info` points to writable storage for the output structure.
    check_status(unsafe { ffi::amdsmi_get_power_info(handle, info.as_mut_ptr()) })?;
    // SAFETY: AMD SMI initialized the output structure after returning success.
    Ok(unsafe { info.assume_init() })
}

fn power_cap_info(
    handle: ffi::AmdsmiProcessorHandle,
) -> Result<ffi::AmdsmiPowerCapInfo, ZeusdError> {
    let mut info = MaybeUninit::zeroed();
    // SAFETY: `info` points to writable storage for the output structure.
    check_status(unsafe { ffi::amdsmi_get_power_cap_info(handle, 0, info.as_mut_ptr()) })?;
    // SAFETY: AMD SMI initialized the output structure after returning success.
    Ok(unsafe { info.assume_init() })
}

fn clock_info(
    handle: ffi::AmdsmiProcessorHandle,
    clock_type: u32,
) -> Result<ffi::AmdsmiClkInfo, ZeusdError> {
    let mut info = MaybeUninit::zeroed();
    // SAFETY: `info` points to writable storage for the output structure.
    check_status(unsafe { ffi::amdsmi_get_clock_info(handle, clock_type, info.as_mut_ptr()) })?;
    // SAFETY: AMD SMI initialized the output structure after returning success.
    Ok(unsafe { info.assume_init() })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WriteOrder {
    MinFirst,
    MaxFirst,
}

fn clk_write_order(cached: Option<(u32, u32)>, new_min: u32, _new_max: u32) -> WriteOrder {
    if cached.is_some_and(|(_, cached_max)| new_min > cached_max) {
        WriteOrder::MaxFirst
    } else {
        WriteOrder::MinFirst
    }
}

impl AmdsmiGpu {
    /// Initialize an AMD SMI handle for a GPU index.
    pub fn init(index: u32) -> Result<Self, ZeusdError> {
        let handles = enumerate_gpus()?;
        let handle = handles
            .get(index as usize)
            .copied()
            .ok_or(ZeusdError::GpuNotFoundError(index as usize))?;
        let power = power_info(handle)?;
        let power_reading = if power.current_socket_power != ffi::AMDSMI_POWER_NA {
            PowerReading::Current
        } else if power.average_socket_power != ffi::AMDSMI_POWER_NA {
            PowerReading::Average
        } else {
            PowerReading::Unsupported
        };
        tracing::info!(
            "AMD SMI GPU {} power reading selected: {:?}",
            index,
            power_reading
        );

        // `clock_info` min/max semantics vary by AMD SMI version; 6.4 excludes the current level from the range, and mis-seeds self-correct through the set path's requery fallback.
        let gfx_clock_bounds = match clock_info(handle, ffi::AMDSMI_CLK_TYPE_GFX) {
            Ok(info) => Some((info.min_clk, info.max_clk)),
            Err(error) => {
                tracing::debug!(
                    "Could not cache GFX clock bounds for AMD SMI GPU {}: {}",
                    index,
                    error
                );
                None
            }
        };
        let mem_clock_bounds = match clock_info(handle, ffi::AMDSMI_CLK_TYPE_MEM) {
            Ok(info) => Some((info.min_clk, info.max_clk)),
            Err(error) => {
                tracing::debug!(
                    "Could not cache MEM clock bounds for AMD SMI GPU {}: {}",
                    index,
                    error
                );
                None
            }
        };

        let gpu = Self {
            handle,
            power_reading,
            gfx_clock_bounds,
            mem_clock_bounds,
        };
        let name = gpu.name()?;
        let bdf = gpu.bdf()?;
        tracing::info!(
            "Initialized AMD SMI for GPU {} ({}, BDF {})",
            index,
            name,
            bdf
        );
        Ok(gpu)
    }

    /// Read the GPU market name.
    pub fn name(&self) -> Result<String, ZeusdError> {
        let mut info = MaybeUninit::zeroed();
        // SAFETY: `info` points to writable storage for the output structure.
        check_status(unsafe { ffi::amdsmi_get_gpu_asic_info(self.handle, info.as_mut_ptr()) })?;
        // SAFETY: AMD SMI initialized the output structure after returning success.
        let info = unsafe { info.assume_init() };
        let end = info
            .market_name
            .iter()
            .position(|&character| character == 0)
            .unwrap_or(info.market_name.len());
        let bytes: Vec<u8> = info.market_name[..end]
            .iter()
            .map(|&character| character as u8)
            .collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Read the GPU PCI bus, device, and function identifier.
    pub fn bdf(&self) -> Result<ffi::Bdf, ZeusdError> {
        let mut bdf = MaybeUninit::zeroed();
        // SAFETY: `bdf` points to writable storage for the output value.
        check_status(unsafe { ffi::amdsmi_get_gpu_device_bdf(self.handle, bdf.as_mut_ptr()) })?;
        // SAFETY: AMD SMI initialized the output value after returning success.
        Ok(unsafe { bdf.assume_init() })
    }

    fn clock_bounds(&self, clock_type: u32) -> Option<(u32, u32)> {
        match clock_type {
            ffi::AMDSMI_CLK_TYPE_GFX => self.gfx_clock_bounds,
            ffi::AMDSMI_CLK_TYPE_MEM => self.mem_clock_bounds,
            _ => unreachable!("unsupported AMD SMI clock type: {clock_type}"),
        }
    }

    fn set_clock_bounds(&mut self, clock_type: u32, bounds: Option<(u32, u32)>) {
        match clock_type {
            ffi::AMDSMI_CLK_TYPE_GFX => self.gfx_clock_bounds = bounds,
            ffi::AMDSMI_CLK_TYPE_MEM => self.mem_clock_bounds = bounds,
            _ => unreachable!("unsupported AMD SMI clock type: {clock_type}"),
        }
    }

    fn write_locked_clocks(
        &self,
        clock_type: u32,
        min_clock_mhz: u32,
        max_clock_mhz: u32,
        order: WriteOrder,
    ) -> Result<(), ZeusdError> {
        let limits = match order {
            WriteOrder::MinFirst => [
                (ffi::AMDSMI_CLK_LIMIT_MIN, min_clock_mhz),
                (ffi::AMDSMI_CLK_LIMIT_MAX, max_clock_mhz),
            ],
            WriteOrder::MaxFirst => [
                (ffi::AMDSMI_CLK_LIMIT_MAX, max_clock_mhz),
                (ffi::AMDSMI_CLK_LIMIT_MIN, min_clock_mhz),
            ],
        };
        for (limit_type, clock_mhz) in limits {
            // SAFETY: `self.handle` is a valid process-wide AMD SMI processor handle.
            check_status(unsafe {
                ffi::amdsmi_set_gpu_clk_limit(
                    self.handle,
                    clock_type,
                    limit_type,
                    u64::from(clock_mhz),
                )
            })?;
        }
        Ok(())
    }

    fn set_locked_clocks(
        &mut self,
        clock_type: u32,
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    ) -> Result<(), ZeusdError> {
        let cached = match self.clock_bounds(clock_type) {
            Some(bounds) => Some(bounds),
            None => {
                let info = clock_info(self.handle, clock_type)?;
                let bounds = (info.min_clk, info.max_clk);
                self.set_clock_bounds(clock_type, Some(bounds));
                Some(bounds)
            }
        };
        let order = clk_write_order(cached, min_clock_mhz, max_clock_mhz);

        match self.write_locked_clocks(clock_type, min_clock_mhz, max_clock_mhz, order) {
            Ok(()) => {
                self.set_clock_bounds(clock_type, Some((min_clock_mhz, max_clock_mhz)));
                return Ok(());
            }
            Err(error) => {
                tracing::warn!(
                    "Failed to set AMD SMI GPU clock type {} range: {}; the GPU's clock range was likely changed outside zeusd, requerying and retrying",
                    clock_type,
                    error
                );
                self.set_clock_bounds(clock_type, None);
            }
        }

        let info = clock_info(self.handle, clock_type)?;
        let queried_bounds = (info.min_clk, info.max_clk);
        let order = clk_write_order(Some(queried_bounds), min_clock_mhz, max_clock_mhz);
        match self.write_locked_clocks(clock_type, min_clock_mhz, max_clock_mhz, order) {
            Ok(()) => {
                self.set_clock_bounds(clock_type, Some((min_clock_mhz, max_clock_mhz)));
                Ok(())
            }
            Err(error) => {
                self.set_clock_bounds(clock_type, None);
                if let Some(bounds) = cached {
                    let revert_order = clk_write_order(Some(queried_bounds), bounds.0, bounds.1);
                    match self.write_locked_clocks(clock_type, bounds.0, bounds.1, revert_order) {
                        Ok(()) => self.set_clock_bounds(clock_type, Some(bounds)),
                        Err(revert_error) => {
                            tracing::warn!(
                                "Failed to restore AMD SMI GPU clock type {} range after set failure: {}",
                                clock_type,
                                revert_error
                            );
                        }
                    }
                }
                Err(error)
            }
        }
    }
}

impl GpuManager for AmdsmiGpu {
    fn device_count() -> Result<u32, ZeusdError> {
        Ok(enumerate_gpus()?.len() as u32)
    }

    fn set_persistence_mode(&mut self, _enabled: bool) -> Result<(), ZeusdError> {
        Err(ZeusdError::InvalidRequest(
            "persistence mode is not supported on AMD GPUs".into(),
        ))
    }

    fn get_persistence_mode(&mut self) -> Result<bool, ZeusdError> {
        Err(ZeusdError::InvalidRequest(
            "persistence mode is not supported on AMD GPUs".into(),
        ))
    }

    fn set_power_management_limit(&mut self, power_limit_mw: u32) -> Result<(), ZeusdError> {
        // SAFETY: `self.handle` is valid and AMD SMI accepts the cap in uW.
        check_status(unsafe {
            ffi::amdsmi_set_power_cap(self.handle, 0, u64::from(power_limit_mw) * 1000)
        })
    }

    fn get_power_management_limit(&mut self) -> Result<u32, ZeusdError> {
        Ok((power_cap_info(self.handle)?.power_cap / 1000) as u32)
    }

    fn get_power_management_limit_constraints(&mut self) -> Result<(u32, u32), ZeusdError> {
        let info = power_cap_info(self.handle)?;
        Ok((
            (info.min_power_cap / 1000) as u32,
            (info.max_power_cap / 1000) as u32,
        ))
    }

    fn set_gpu_locked_clocks(
        &mut self,
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    ) -> Result<(), ZeusdError> {
        self.set_locked_clocks(ffi::AMDSMI_CLK_TYPE_GFX, min_clock_mhz, max_clock_mhz)
    }

    fn reset_gpu_locked_clocks(&mut self) -> Result<(), ZeusdError> {
        Err(ZeusdError::InvalidRequest(
            "AMD GPUs cannot reset a single clock domain; use reset_locked_clocks to reset all domains"
                .into(),
        ))
    }

    fn set_mem_locked_clocks(
        &mut self,
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    ) -> Result<(), ZeusdError> {
        self.set_locked_clocks(ffi::AMDSMI_CLK_TYPE_MEM, min_clock_mhz, max_clock_mhz)
    }

    fn reset_mem_locked_clocks(&mut self) -> Result<(), ZeusdError> {
        Err(ZeusdError::InvalidRequest(
            "AMD GPUs cannot reset a single clock domain; use reset_locked_clocks to reset all domains"
                .into(),
        ))
    }

    /// AMD offers no per-domain reset, so this restores the driver's automatic
    /// frequency management for all clock domains and lets the kernel restore
    /// default limits.
    fn reset_locked_clocks(&mut self) -> Result<(), ZeusdError> {
        // SAFETY: `self.handle` is a valid process-wide AMD SMI processor handle.
        check_status(unsafe {
            ffi::amdsmi_set_gpu_perf_level(self.handle, ffi::AMDSMI_DEV_PERF_LEVEL_AUTO)
        })?;
        self.gfx_clock_bounds = None;
        self.mem_clock_bounds = None;
        Ok(())
    }

    fn get_instant_power_mw(&mut self) -> Result<u32, ZeusdError> {
        let info = power_info(self.handle)?;
        let power_w = match self.power_reading {
            PowerReading::Current => info.current_socket_power,
            PowerReading::Average => info.average_socket_power,
            PowerReading::Unsupported => {
                return Err(amdsmi_error(ffi::AMDSMI_STATUS_NOT_SUPPORTED));
            }
        };
        if power_w == ffi::AMDSMI_POWER_NA {
            return Err(amdsmi_error(ffi::AMDSMI_STATUS_NOT_SUPPORTED));
        }
        Ok(power_w * 1000)
    }

    fn get_total_energy_consumption(&mut self) -> Result<u64, ZeusdError> {
        let mut energy_accumulator = 0;
        let mut counter_resolution = 0.0;
        let mut timestamp = 0;
        // SAFETY: All output pointers refer to writable values of the expected types.
        check_status(unsafe {
            ffi::amdsmi_get_energy_count(
                self.handle,
                &mut energy_accumulator,
                &mut counter_resolution,
                &mut timestamp,
            )
        })?;
        Ok((energy_accumulator as f64 * counter_resolution as f64 / 1000.0) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clk_write_order_raises_both_max_first() {
        assert_eq!(
            clk_write_order(Some((1_000, 1_500)), 1_600, 1_800),
            WriteOrder::MaxFirst
        );
    }

    #[test]
    fn clk_write_order_lowers_both_min_first() {
        assert_eq!(
            clk_write_order(Some((1_000, 1_500)), 500, 900),
            WriteOrder::MinFirst
        );
    }

    #[test]
    fn clk_write_order_overlapping_range_min_first() {
        assert_eq!(
            clk_write_order(Some((1_000, 1_500)), 1_200, 1_800),
            WriteOrder::MinFirst
        );
    }

    #[test]
    fn clk_write_order_without_cache_min_first() {
        assert_eq!(clk_write_order(None, 1_600, 1_800), WriteOrder::MinFirst);
    }

    #[test]
    fn clk_write_order_equal_cached_max_min_first() {
        assert_eq!(
            clk_write_order(Some((1_000, 1_500)), 1_500, 1_800),
            WriteOrder::MinFirst
        );
    }
}
