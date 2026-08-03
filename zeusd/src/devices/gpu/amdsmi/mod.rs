//! Implements `AmdsmiGpu`, a GPU manager for AMD GPUs using AMD SMI.
//!
//! AMD SMI is available on Linux as part of ROCm. The library is initialized
//! once and remains initialized for the lifetime of the daemon.

use std::env;
use std::ffi::{CStr, OsStr};
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::ptr;

use libloading::Library;
use once_cell::sync::OnceCell;

use crate::devices::gpu::GpuManager;
use crate::error::ZeusdError;

pub mod ffi;

static AMDSMI_API: OnceCell<Result<AmdsmiApi, String>> = OnceCell::new();
static AMDSMI_INIT_STATUS: OnceCell<ffi::AmdsmiStatus> = OnceCell::new();
static AMDSMI_VERSION_VALIDATION: OnceCell<Result<(), String>> = OnceCell::new();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AmdsmiAbi {
    V24,
    V25,
    V26,
}

impl AmdsmiAbi {
    const SUPPORTED: [(Self, &'static str); 3] = [
        (Self::V26, "libamd_smi.so.26"),
        (Self::V25, "libamd_smi.so.25"),
        (Self::V24, "libamd_smi.so.24"),
    ];

    fn number(self) -> u32 {
        match self {
            Self::V24 => 24,
            Self::V25 => 25,
            Self::V26 => 26,
        }
    }
}

struct AmdsmiApi {
    _library: Library,
    library_path: PathBuf,
    abi: AmdsmiAbi,
    init: ffi::AmdsmiInitFn,
    _shut_down: ffi::AmdsmiShutDownFn,
    get_socket_handles: ffi::AmdsmiGetSocketHandlesFn,
    get_processor_handles: ffi::AmdsmiGetProcessorHandlesFn,
    get_gpu_asic_info: ffi::AmdsmiGetGpuAsicInfoFn,
    get_gpu_device_bdf: ffi::AmdsmiGetGpuDeviceBdfFn,
    get_power_cap_info: ffi::AmdsmiGetPowerCapInfoFn,
    set_power_cap: ffi::AmdsmiSetPowerCapFn,
    get_power_info: ffi::AmdsmiGetPowerInfoFn,
    get_energy_count: ffi::AmdsmiGetEnergyCountFn,
    get_clock_info: ffi::AmdsmiGetClockInfoFn,
    get_lib_version: ffi::AmdsmiGetLibVersionFn,
    status_code_to_string: ffi::AmdsmiStatusCodeToStringFn,
    set_gpu_clk_limit: ffi::AmdsmiSetGpuClkLimitFn,
    set_gpu_perf_level: ffi::AmdsmiSetGpuPerfLevelFn,
}

#[derive(Debug, Eq, PartialEq)]
struct AmdsmiLibraryVersion {
    year: Option<u32>,
    major: u32,
    minor: u32,
    release: u32,
}

impl std::fmt::Display for AmdsmiLibraryVersion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(year) = self.year {
            write!(
                formatter,
                "{}.{}.{}.{}",
                year, self.major, self.minor, self.release
            )
        } else {
            write!(formatter, "{}.{}.{}", self.major, self.minor, self.release)
        }
    }
}

impl AmdsmiApi {
    fn load() -> Result<Self, String> {
        let candidates = library_candidates(
            env::var_os("AMDSMI_LIB_DIR").as_deref(),
            env::var_os("ROCM_PATH").as_deref(),
        );
        let mut errors = Vec::with_capacity(candidates.len());
        for (abi, path) in candidates {
            // SAFETY: The returned library is retained by `AmdsmiApi` for as long as any loaded symbol can be called.
            match unsafe { Library::new(&path) } {
                Ok(library) => match Self::from_library(library, abi, &path) {
                    Ok(api) => {
                        tracing::info!(
                            "Loaded AMD SMI from {} using ABI {}",
                            path.display(),
                            abi.number()
                        );
                        return Ok(api);
                    }
                    // A library that opens but fails symbol resolution must not
                    // abort the search; a later candidate can still be usable.
                    Err(error) => {
                        // ROCm 6.2 ships libamd_smi.so.24 without this symbol, so
                        // resolution fails before the version floor check can run.
                        if abi == AmdsmiAbi::V24 && error.contains("amdsmi_set_gpu_clk_limit") {
                            errors.push(format!(
                                "{error}; a libamd_smi.so.24 without this symbol is ROCm 6.2, \
                                 which zeusd does not support (ROCm 6.3 is the minimum)"
                            ));
                        } else {
                            errors.push(error);
                        }
                    }
                },
                Err(error) => errors.push(format!("{}: {}", path.display(), error)),
            }
        }
        Err(format!(
            "no supported AMD SMI library found (AMDSMI_LIB_DIR and ROCM_PATH restrict the search when set; otherwise /opt/rocm*/lib and the dynamic loader paths are searched): {}",
            errors.join("; ")
        ))
    }

    fn from_library(library: Library, abi: AmdsmiAbi, path: &Path) -> Result<Self, String> {
        // SAFETY: Each type alias matches the corresponding AMD SMI C declaration for all supported ABIs.
        unsafe {
            Ok(Self {
                init: load_symbol(&library, b"amdsmi_init\0", path)?,
                _shut_down: load_symbol(&library, b"amdsmi_shut_down\0", path)?,
                get_socket_handles: load_symbol(&library, b"amdsmi_get_socket_handles\0", path)?,
                get_processor_handles: load_symbol(
                    &library,
                    b"amdsmi_get_processor_handles\0",
                    path,
                )?,
                get_gpu_asic_info: load_symbol(&library, b"amdsmi_get_gpu_asic_info\0", path)?,
                get_gpu_device_bdf: load_symbol(&library, b"amdsmi_get_gpu_device_bdf\0", path)?,
                get_power_cap_info: load_symbol(&library, b"amdsmi_get_power_cap_info\0", path)?,
                set_power_cap: load_symbol(&library, b"amdsmi_set_power_cap\0", path)?,
                get_power_info: load_symbol(&library, b"amdsmi_get_power_info\0", path)?,
                get_energy_count: load_symbol(&library, b"amdsmi_get_energy_count\0", path)?,
                get_clock_info: load_symbol(&library, b"amdsmi_get_clock_info\0", path)?,
                get_lib_version: load_symbol(&library, b"amdsmi_get_lib_version\0", path)?,
                status_code_to_string: load_symbol(
                    &library,
                    b"amdsmi_status_code_to_string\0",
                    path,
                )?,
                set_gpu_clk_limit: load_symbol(&library, b"amdsmi_set_gpu_clk_limit\0", path)?,
                set_gpu_perf_level: load_symbol(&library, b"amdsmi_set_gpu_perf_level\0", path)?,
                _library: library,
                library_path: path.to_owned(),
                abi,
            })
        }
    }

    fn library_version(&self) -> Result<AmdsmiLibraryVersion, String> {
        match self.abi {
            AmdsmiAbi::V24 | AmdsmiAbi::V25 => {
                let mut version = MaybeUninit::<ffi::AmdsmiVersion24And25>::zeroed();
                // SAFETY: The ABI-specific output buffer matches the loaded library's SONAME.
                let status = unsafe { (self.get_lib_version)(version.as_mut_ptr().cast()) };
                self.check_version_status(status)?;
                // SAFETY: AMD SMI initialized the output structure after returning success.
                let version = unsafe { version.assume_init() };
                Ok(AmdsmiLibraryVersion {
                    year: Some(version.year),
                    major: version.major,
                    minor: version.minor,
                    release: version.release,
                })
            }
            AmdsmiAbi::V26 => {
                let mut version = MaybeUninit::<ffi::AmdsmiVersion26>::zeroed();
                // SAFETY: The ABI-specific output buffer matches the loaded library's SONAME.
                let status = unsafe { (self.get_lib_version)(version.as_mut_ptr().cast()) };
                self.check_version_status(status)?;
                // SAFETY: AMD SMI initialized the output structure after returning success.
                let version = unsafe { version.assume_init() };
                Ok(AmdsmiLibraryVersion {
                    year: None,
                    major: version.major,
                    minor: version.minor,
                    release: version.release,
                })
            }
        }
    }

    fn check_version_status(&self, status: ffi::AmdsmiStatus) -> Result<(), String> {
        if status != ffi::AMDSMI_STATUS_SUCCESS {
            Err(format!(
                "amdsmi_get_lib_version failed: {}",
                self.status_message(status)
            ))
        } else {
            Ok(())
        }
    }

    fn status_message(&self, status: ffi::AmdsmiStatus) -> String {
        let mut message_ptr = ptr::null();
        // SAFETY: AMD SMI writes a borrowed pointer to a static status string.
        let message_status = unsafe { (self.status_code_to_string)(status, &mut message_ptr) };
        if message_status == ffi::AMDSMI_STATUS_SUCCESS && !message_ptr.is_null() {
            // SAFETY: A successful call returns a valid NUL-terminated string.
            unsafe { CStr::from_ptr(message_ptr) }
                .to_string_lossy()
                .into_owned()
        } else {
            status.to_string()
        }
    }

    fn power_info(
        &self,
        handle: ffi::AmdsmiProcessorHandle,
    ) -> Result<AmdsmiPowerReadings, ZeusdError> {
        match self.abi {
            AmdsmiAbi::V24 => {
                let mut info = MaybeUninit::<ffi::AmdsmiPowerInfo24>::zeroed();
                // SAFETY: The ABI-specific output buffer matches the loaded library's SONAME.
                let status = unsafe { (self.get_power_info)(handle, info.as_mut_ptr().cast()) };
                check_status(self, status)?;
                // SAFETY: AMD SMI initialized the output structure after returning success.
                let info = unsafe { info.assume_init() };
                Ok(AmdsmiPowerReadings::from_24(info))
            }
            AmdsmiAbi::V25 => {
                let mut info = MaybeUninit::<ffi::AmdsmiPowerInfo25>::zeroed();
                // SAFETY: The ABI-specific output buffer matches the loaded library's SONAME.
                let status = unsafe { (self.get_power_info)(handle, info.as_mut_ptr().cast()) };
                check_status(self, status)?;
                // SAFETY: AMD SMI initialized the output structure after returning success.
                let info = unsafe { info.assume_init() };
                Ok(AmdsmiPowerReadings::from_25(info))
            }
            AmdsmiAbi::V26 => {
                let mut info = MaybeUninit::<ffi::AmdsmiPowerInfo26>::zeroed();
                // SAFETY: The ABI-specific output buffer matches the loaded library's SONAME.
                let status = unsafe { (self.get_power_info)(handle, info.as_mut_ptr().cast()) };
                check_status(self, status)?;
                // SAFETY: AMD SMI initialized the output structure after returning success.
                let info = unsafe { info.assume_init() };
                Ok(AmdsmiPowerReadings::from_26(info))
            }
        }
    }
}

fn validate_library_version(abi: AmdsmiAbi, version: &AmdsmiLibraryVersion) -> Result<(), String> {
    let valid = match abi {
        AmdsmiAbi::V24 => version.year == Some(24) && version.major >= 7,
        AmdsmiAbi::V25 => version.year == Some(25) && version.major == 25,
        AmdsmiAbi::V26 => version.year.is_none() && version.major == 26,
    };
    if valid {
        return Ok(());
    }
    if abi == AmdsmiAbi::V24 && version.year == Some(24) && version.major < 7 {
        return Err(format!(
            "found AMD SMI {} from ROCm 6.2; the oldest supported ROCm is 6.3",
            version
        ));
    }
    Err(format!(
        "libamd_smi.so.{} reported incompatible library version {}",
        abi.number(),
        version
    ))
}

unsafe fn load_symbol<T: Copy>(library: &Library, symbol: &[u8], path: &Path) -> Result<T, String> {
    // SAFETY: The caller supplies the exact function-pointer type for the named C symbol.
    unsafe { library.get::<T>(symbol) }
        .map(|loaded| *loaded)
        .map_err(|error| {
            let name = String::from_utf8_lossy(symbol)
                .trim_end_matches('\0')
                .to_owned();
            format!("failed to load {name} from {}: {error}", path.display())
        })
}

/// Parse the numeric version components of an `/opt/rocm-<version>` directory
/// name so that, e.g., `rocm-6.10.0` orders after `rocm-6.9.0`.
fn rocm_dir_version(path: &Path) -> Option<Vec<u32>> {
    path.file_name()?
        .to_str()?
        .strip_prefix("rocm-")?
        .split('.')
        .map(|part| part.parse().ok())
        .collect()
}

fn newest_opt_rocm_lib() -> Option<PathBuf> {
    std::fs::read_dir("/opt")
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .filter_map(|path| rocm_dir_version(&path).map(|version| (version, path)))
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, path)| path.join("lib"))
}

/// Build the ordered list of library paths to try. `AMDSMI_LIB_DIR` and
/// `ROCM_PATH` are authoritative when set: the search is restricted to the
/// named installation instead of falling back to default locations, so a
/// misconfigured override fails loudly rather than silently using another
/// library.
fn library_candidates(
    lib_dir: Option<&OsStr>,
    rocm_path: Option<&OsStr>,
) -> Vec<(AmdsmiAbi, PathBuf)> {
    let mut roots = Vec::new();
    let mut search_loader_paths = false;
    if let Some(lib_dir) = lib_dir {
        roots.push(PathBuf::from(lib_dir));
    } else if let Some(rocm_path) = rocm_path {
        roots.push(PathBuf::from(rocm_path).join("lib"));
    } else {
        roots.push(PathBuf::from("/opt/rocm/lib"));
        if let Some(newest) = newest_opt_rocm_lib() {
            if !roots.contains(&newest) {
                roots.push(newest);
            }
        }
        search_loader_paths = true;
    }

    let mut candidates = Vec::new();
    for root in roots {
        candidates.extend(
            AmdsmiAbi::SUPPORTED
                .into_iter()
                .map(|(abi, name)| (abi, root.join(name))),
        );
    }
    if search_loader_paths {
        candidates.extend(
            AmdsmiAbi::SUPPORTED
                .into_iter()
                .map(|(abi, name)| (abi, PathBuf::from(name))),
        );
    }
    candidates
}

fn amdsmi_api() -> Result<&'static AmdsmiApi, ZeusdError> {
    AMDSMI_API
        .get_or_init(AmdsmiApi::load)
        .as_ref()
        .map_err(|error| ZeusdError::AmdSmiLoadError(error.clone()))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AmdsmiPowerReadings {
    current_socket_power: u32,
    average_socket_power: u32,
}

impl AmdsmiPowerReadings {
    fn from_24(info: ffi::AmdsmiPowerInfo24) -> Self {
        Self {
            current_socket_power: info.current_socket_power,
            average_socket_power: info.average_socket_power,
        }
    }

    fn from_25(info: ffi::AmdsmiPowerInfo25) -> Self {
        Self {
            current_socket_power: info.current_socket_power,
            average_socket_power: info.average_socket_power,
        }
    }

    fn from_26(info: ffi::AmdsmiPowerInfo26) -> Self {
        Self {
            current_socket_power: info.current_socket_power,
            average_socket_power: info.average_socket_power,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum PowerReading {
    Current,
    Average,
    Unsupported,
}

/// A GPU managed through AMD SMI.
pub struct AmdsmiGpu {
    api: &'static AmdsmiApi,
    handle: ffi::AmdsmiProcessorHandle,
    power_reading: PowerReading,
    // Single-task ownership serializes cache updates; failed writes requery bounds to cover external changes.
    gfx_clock_bounds: Option<(u32, u32)>,
    mem_clock_bounds: Option<(u32, u32)>,
}

// SAFETY: Processor handles are process-wide identifiers into the thread-safe
// AMD SMI library, not thread-local resources.
unsafe impl Send for AmdsmiGpu {}

fn amdsmi_error(api: &AmdsmiApi, status: ffi::AmdsmiStatus) -> ZeusdError {
    ZeusdError::AmdSmiError {
        status,
        msg: api.status_message(status),
    }
}

fn check_status(api: &AmdsmiApi, status: ffi::AmdsmiStatus) -> Result<(), ZeusdError> {
    if status == ffi::AMDSMI_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(amdsmi_error(api, status))
    }
}

fn init_amdsmi(api: &AmdsmiApi) -> Result<(), ZeusdError> {
    let status = AMDSMI_INIT_STATUS.get_or_init(|| {
        // SAFETY: This process-global call is executed exactly once by `OnceCell`.
        unsafe { (api.init)(ffi::AMDSMI_INIT_AMD_GPUS) }
    });
    check_status(api, *status)?;
    AMDSMI_VERSION_VALIDATION
        .get_or_init(|| {
            let version = api.library_version()?;
            validate_library_version(api.abi, &version)?;
            tracing::info!(
                "Initialized AMD SMI {} from {} using ABI {}",
                version,
                api.library_path.display(),
                api.abi.number()
            );
            Ok(())
        })
        .as_ref()
        .map_err(|error| ZeusdError::AmdSmiLoadError(error.clone()))
        .copied()
}

fn socket_handles(api: &AmdsmiApi) -> Result<Vec<ffi::AmdsmiSocketHandle>, ZeusdError> {
    let mut count = 0;
    // SAFETY: A null buffer requests the required socket count.
    check_status(api, unsafe {
        (api.get_socket_handles)(&mut count, ptr::null_mut())
    })?;
    let mut handles = vec![ptr::null_mut(); count as usize];
    if count > 0 {
        // SAFETY: `handles` has room for the count returned by the first call.
        check_status(api, unsafe {
            (api.get_socket_handles)(&mut count, handles.as_mut_ptr())
        })?;
        handles.truncate(count as usize);
    }
    Ok(handles)
}

fn processor_handles(
    api: &AmdsmiApi,
    socket: ffi::AmdsmiSocketHandle,
) -> Result<Vec<ffi::AmdsmiProcessorHandle>, ZeusdError> {
    let mut count = 0;
    // SAFETY: A null buffer requests the required processor count.
    check_status(api, unsafe {
        (api.get_processor_handles)(socket, &mut count, ptr::null_mut())
    })?;
    let mut handles = vec![ptr::null_mut(); count as usize];
    if count > 0 {
        // SAFETY: `handles` has room for the count returned by the first call.
        check_status(api, unsafe {
            (api.get_processor_handles)(socket, &mut count, handles.as_mut_ptr())
        })?;
        handles.truncate(count as usize);
    }
    Ok(handles)
}

fn enumerate_gpus(api: &AmdsmiApi) -> Result<Vec<ffi::AmdsmiProcessorHandle>, ZeusdError> {
    init_amdsmi(api)?;
    let mut handles = Vec::new();
    for socket in socket_handles(api)? {
        handles.extend(processor_handles(api, socket)?);
    }
    Ok(handles)
}

fn power_cap_info(
    api: &AmdsmiApi,
    handle: ffi::AmdsmiProcessorHandle,
) -> Result<ffi::AmdsmiPowerCapInfo, ZeusdError> {
    let mut info = MaybeUninit::zeroed();
    // SAFETY: `info` points to writable storage for the output structure.
    check_status(api, unsafe {
        (api.get_power_cap_info)(handle, 0, info.as_mut_ptr())
    })?;
    // SAFETY: AMD SMI initialized the output structure after returning success.
    Ok(unsafe { info.assume_init() })
}

fn clock_info(
    api: &AmdsmiApi,
    handle: ffi::AmdsmiProcessorHandle,
    clock_type: u32,
) -> Result<ffi::AmdsmiClkInfo, ZeusdError> {
    let mut info = MaybeUninit::zeroed();
    // SAFETY: `info` points to writable storage for the output structure.
    check_status(api, unsafe {
        (api.get_clock_info)(handle, clock_type, info.as_mut_ptr())
    })?;
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
        let api = amdsmi_api()?;
        let handles = enumerate_gpus(api)?;
        let handle = handles
            .get(index as usize)
            .copied()
            .ok_or(ZeusdError::GpuNotFoundError(index as usize))?;
        let power = api.power_info(handle)?;
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
        let gfx_clock_bounds = match clock_info(api, handle, ffi::AMDSMI_CLK_TYPE_GFX) {
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
        let mem_clock_bounds = match clock_info(api, handle, ffi::AMDSMI_CLK_TYPE_MEM) {
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
            api,
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
        check_status(self.api, unsafe {
            (self.api.get_gpu_asic_info)(self.handle, info.as_mut_ptr())
        })?;
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
        check_status(self.api, unsafe {
            (self.api.get_gpu_device_bdf)(self.handle, bdf.as_mut_ptr())
        })?;
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
            check_status(self.api, unsafe {
                (self.api.set_gpu_clk_limit)(
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
                let info = clock_info(self.api, self.handle, clock_type)?;
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

        let info = clock_info(self.api, self.handle, clock_type)?;
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
        let api = amdsmi_api()?;
        Ok(enumerate_gpus(api)?.len() as u32)
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
        check_status(self.api, unsafe {
            (self.api.set_power_cap)(self.handle, 0, u64::from(power_limit_mw) * 1000)
        })
    }

    fn get_power_management_limit(&mut self) -> Result<u32, ZeusdError> {
        Ok((power_cap_info(self.api, self.handle)?.power_cap / 1000) as u32)
    }

    fn get_power_management_limit_constraints(&mut self) -> Result<(u32, u32), ZeusdError> {
        let info = power_cap_info(self.api, self.handle)?;
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
        check_status(self.api, unsafe {
            (self.api.set_gpu_perf_level)(self.handle, ffi::AMDSMI_DEV_PERF_LEVEL_AUTO)
        })?;
        self.gfx_clock_bounds = None;
        self.mem_clock_bounds = None;
        Ok(())
    }

    fn get_instant_power_mw(&mut self) -> Result<u32, ZeusdError> {
        let info = self.api.power_info(self.handle)?;
        let power_w = match self.power_reading {
            PowerReading::Current => info.current_socket_power,
            PowerReading::Average => info.average_socket_power,
            PowerReading::Unsupported => {
                return Err(amdsmi_error(self.api, ffi::AMDSMI_STATUS_NOT_SUPPORTED));
            }
        };
        if power_w == ffi::AMDSMI_POWER_NA {
            return Err(amdsmi_error(self.api, ffi::AMDSMI_STATUS_NOT_SUPPORTED));
        }
        Ok(power_w * 1000)
    }

    fn get_total_energy_consumption(&mut self) -> Result<u64, ZeusdError> {
        let mut energy_accumulator = 0;
        let mut counter_resolution = 0.0;
        let mut timestamp = 0;
        // SAFETY: All output pointers refer to writable values of the expected types.
        check_status(self.api, unsafe {
            (self.api.get_energy_count)(
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
    fn lib_dir_override_restricts_the_search() {
        let candidates = library_candidates(
            Some(OsStr::new("/custom/libs")),
            Some(OsStr::new("/custom/rocm")),
        );
        assert_eq!(
            candidates,
            [
                (
                    AmdsmiAbi::V26,
                    PathBuf::from("/custom/libs/libamd_smi.so.26")
                ),
                (
                    AmdsmiAbi::V25,
                    PathBuf::from("/custom/libs/libamd_smi.so.25")
                ),
                (
                    AmdsmiAbi::V24,
                    PathBuf::from("/custom/libs/libamd_smi.so.24")
                ),
            ]
        );
    }

    #[test]
    fn rocm_path_override_restricts_the_search() {
        let candidates = library_candidates(None, Some(OsStr::new("/custom/rocm")));
        assert_eq!(
            candidates,
            [
                (
                    AmdsmiAbi::V26,
                    PathBuf::from("/custom/rocm/lib/libamd_smi.so.26")
                ),
                (
                    AmdsmiAbi::V25,
                    PathBuf::from("/custom/rocm/lib/libamd_smi.so.25")
                ),
                (
                    AmdsmiAbi::V24,
                    PathBuf::from("/custom/rocm/lib/libamd_smi.so.24")
                ),
            ]
        );
    }

    #[test]
    fn rocm_dir_versions_compare_numerically() {
        assert!(
            rocm_dir_version(Path::new("/opt/rocm-6.10.0")).unwrap()
                > rocm_dir_version(Path::new("/opt/rocm-6.9.0")).unwrap()
        );
        assert!(rocm_dir_version(Path::new("/opt/rocm-custom")).is_none());
    }

    #[test]
    fn supported_library_versions_match_their_abis() {
        assert!(validate_library_version(
            AmdsmiAbi::V24,
            &AmdsmiLibraryVersion {
                year: Some(24),
                major: 7,
                minor: 1,
                release: 0,
            }
        )
        .is_ok());
        assert!(validate_library_version(
            AmdsmiAbi::V25,
            &AmdsmiLibraryVersion {
                year: Some(25),
                major: 25,
                minor: 4,
                release: 0,
            }
        )
        .is_ok());
        assert!(validate_library_version(
            AmdsmiAbi::V26,
            &AmdsmiLibraryVersion {
                year: None,
                major: 26,
                minor: 2,
                release: 1,
            }
        )
        .is_ok());
    }

    #[test]
    fn abi_24_rejects_rocm_6_2() {
        let error = validate_library_version(
            AmdsmiAbi::V24,
            &AmdsmiLibraryVersion {
                year: Some(24),
                major: 6,
                minor: 0,
                release: 0,
            },
        )
        .unwrap_err();
        assert!(error.contains("oldest supported ROCm is 6.3"));
    }

    #[test]
    fn power_info_dispatch_preserves_common_fields() {
        let mut info_24 = unsafe { std::mem::zeroed::<ffi::AmdsmiPowerInfo24>() };
        info_24.current_socket_power = 100;
        info_24.average_socket_power = 90;
        let mut info_25 = unsafe { std::mem::zeroed::<ffi::AmdsmiPowerInfo25>() };
        info_25.current_socket_power = 110;
        info_25.average_socket_power = 95;
        let mut info_26 = unsafe { std::mem::zeroed::<ffi::AmdsmiPowerInfo26>() };
        info_26.current_socket_power = 120;
        info_26.average_socket_power = 105;

        assert_eq!(
            AmdsmiPowerReadings::from_24(info_24),
            AmdsmiPowerReadings {
                current_socket_power: 100,
                average_socket_power: 90,
            }
        );
        assert_eq!(
            AmdsmiPowerReadings::from_25(info_25),
            AmdsmiPowerReadings {
                current_socket_power: 110,
                average_socket_power: 95,
            }
        );
        assert_eq!(
            AmdsmiPowerReadings::from_26(info_26),
            AmdsmiPowerReadings {
                current_socket_power: 120,
                average_socket_power: 105,
            }
        );
    }

    #[test]
    #[ignore = "requires an installed AMD SMI library"]
    fn loads_supported_library_and_symbols() {
        AmdsmiApi::load().unwrap();
    }

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
