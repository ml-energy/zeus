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

impl AmdsmiGpu {
    /// Initialize an AMD SMI handle for a GPU index.
    pub fn init(index: u32) -> Result<Self, ZeusdError> {
        let handles = enumerate_gpus()?;
        let handle = handles
            .get(index as usize)
            .copied()
            .ok_or(ZeusdError::GpuNotFoundError(index as usize))?;
        let power = power_info(handle)?;
        let power_reading = if power.current_socket_power != u32::MAX {
            PowerReading::Current
        } else if power.average_socket_power != u32::MAX {
            PowerReading::Average
        } else {
            PowerReading::Unsupported
        };
        tracing::info!(
            "AMD SMI GPU {} power reading selected: {:?}",
            index,
            power_reading
        );

        let gpu = Self {
            handle,
            power_reading,
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

    fn set_locked_clocks(
        &self,
        clock_type: u32,
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    ) -> Result<(), ZeusdError> {
        // SAFETY: `self.handle` is a valid process-wide AMD SMI processor handle.
        check_status(unsafe {
            ffi::amdsmi_set_gpu_clk_limit(
                self.handle,
                clock_type,
                ffi::AMDSMI_CLK_LIMIT_MIN,
                u64::from(min_clock_mhz),
            )
        })?;
        // SAFETY: `self.handle` is a valid process-wide AMD SMI processor handle.
        check_status(unsafe {
            ffi::amdsmi_set_gpu_clk_limit(
                self.handle,
                clock_type,
                ffi::AMDSMI_CLK_LIMIT_MAX,
                u64::from(max_clock_mhz),
            )
        })
    }

    fn reset_locked_clocks(&self, clock_type: u32) -> Result<(), ZeusdError> {
        let info = clock_info(self.handle, clock_type)?;
        self.set_locked_clocks(clock_type, info.min_clk, info.max_clk)
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
        self.reset_locked_clocks(ffi::AMDSMI_CLK_TYPE_GFX)
    }

    fn set_mem_locked_clocks(
        &mut self,
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    ) -> Result<(), ZeusdError> {
        self.set_locked_clocks(ffi::AMDSMI_CLK_TYPE_MEM, min_clock_mhz, max_clock_mhz)
    }

    fn reset_mem_locked_clocks(&mut self) -> Result<(), ZeusdError> {
        self.reset_locked_clocks(ffi::AMDSMI_CLK_TYPE_MEM)
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
        if power_w == u32::MAX {
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
