//! Startup logic.

use actix_web::dev::Server;
use actix_web::{web, App, HttpServer};
#[cfg(unix)]
use anyhow::Context;
use std::collections::HashSet;
#[cfg(unix)]
use std::fs;
use std::net::TcpListener;
#[cfg(unix)]
use std::os::unix::fs::{chown, PermissionsExt};
#[cfg(unix)]
use std::os::unix::net::UnixListener;
#[cfg(unix)]
use std::path::Path;
use tracing::subscriber::set_global_default;
use tracing_log::LogTracer;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{EnvFilter, Registry};

use crate::auth::{AuthMiddleware, SigningKeyData};
use crate::config::ApiGroup;
#[cfg(target_os = "linux")]
use crate::devices::cpu::power::start_cpu_poller;
use crate::devices::cpu::power::{CpuPowerBroadcast, CpuPowerPoller};
use crate::devices::cpu::CpuManagementTasks;
#[cfg(target_os = "linux")]
use crate::devices::cpu::{CpuManager, RaplCpu};
use crate::devices::gpu::power::{start_gpu_poller, GpuPowerBroadcast, GpuPowerPoller};
use crate::devices::gpu::{GpuManagementTasks, GpuManager, NvmlGpu};
use crate::routes::cpu_routes;
use crate::routes::{gpu_control_routes, gpu_read_routes, server_routes, DiscoveryInfo};

/// Initialize tracing with the given where to write logs to.
pub fn init_tracing<S>(sink: S) -> anyhow::Result<()>
where
    S: for<'a> MakeWriter<'a> + Send + Sync + 'static,
{
    LogTracer::init()?;

    let formatter = tracing_subscriber::fmt::layer().with_writer(sink);
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let subscriber = Registry::default().with(formatter).with(env_filter);
    set_global_default(subscriber)?;

    Ok(())
}

/// Create a socket at the given path and bind a UnixListener to it.
#[cfg(unix)]
pub fn get_unix_listener(
    socket_path: &str,
    permissions: u32,
    uid: Option<u32>,
    gid: Option<u32>,
) -> anyhow::Result<UnixListener> {
    if fs::metadata(socket_path).is_ok() {
        tracing::error!(
            "Socket file {} already exists. Please remove it and restart Zeusd.",
            socket_path,
        );
        anyhow::bail!("Socket file already exists");
    }
    if let Some(parent) = Path::new(socket_path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create parent directory for socket {socket_path}")
            })?;
        }
    }
    let listener = UnixListener::bind(socket_path)?;
    fs::set_permissions(socket_path, fs::Permissions::from_mode(permissions))?;
    chown(socket_path, uid, gid)?;
    Ok(listener)
}

/// Initialize NVML and start GPU management tasks.
pub fn start_gpu_device_tasks() -> anyhow::Result<GpuManagementTasks> {
    tracing::info!("Starting NVML and GPU management tasks.");
    let num_gpus = NvmlGpu::device_count()?;
    let mut gpus = Vec::with_capacity(num_gpus as usize);
    for gpu_id in 0..num_gpus {
        let gpu = NvmlGpu::init(gpu_id)?;
        tracing::info!("Initialized NVML for GPU {}", gpu_id);
        gpus.push(gpu);
    }
    Ok(GpuManagementTasks::start(gpus)?)
}

/// Initialize a separate set of NVML handles and start the GPU power poller.
pub fn start_gpu_power_poller(poll_hz: u32) -> anyhow::Result<GpuPowerPoller> {
    tracing::info!("Starting GPU power poller at {} Hz.", poll_hz);
    let num_gpus = NvmlGpu::device_count()?;
    let mut gpus = Vec::with_capacity(num_gpus as usize);
    for gpu_id in 0..num_gpus {
        let gpu = NvmlGpu::init(gpu_id)?;
        gpus.push((gpu_id as usize, gpu));
    }
    Ok(start_gpu_poller(gpus, poll_hz))
}

/// Initialize RAPL and start CPU management tasks.
///
/// Returns the management tasks and a per-CPU DRAM availability vector.
/// RAPL is Linux-specific; on other platforms this errors out.
#[cfg(target_os = "linux")]
pub fn start_cpu_device_tasks() -> anyhow::Result<(CpuManagementTasks, Vec<bool>)> {
    tracing::info!("Starting Rapl and CPU management tasks.");
    let num_cpus = RaplCpu::device_count()?;
    let mut cpus = Vec::with_capacity(num_cpus);
    let mut dram_available = Vec::with_capacity(num_cpus);
    for cpu_id in 0..num_cpus {
        let cpu = RaplCpu::init(cpu_id)?;
        dram_available.push(cpu.is_dram_available());
        tracing::info!(
            "Initialized RAPL for CPU {} (DRAM: {})",
            cpu_id,
            dram_available[cpu_id],
        );
        cpus.push(cpu);
    }
    Ok((CpuManagementTasks::start(cpus)?, dram_available))
}

#[cfg(not(target_os = "linux"))]
pub fn start_cpu_device_tasks() -> anyhow::Result<(CpuManagementTasks, Vec<bool>)> {
    anyhow::bail!(
        "CPU RAPL monitoring is only available on Linux. \
         Remove 'cpu-read' from --enable to start zeusd on this platform."
    )
}

/// Initialize a separate set of RAPL handles and start the CPU power poller.
#[cfg(target_os = "linux")]
pub fn start_cpu_power_poller(poll_hz: u32) -> anyhow::Result<CpuPowerPoller> {
    tracing::info!("Starting CPU RAPL power poller at {} Hz.", poll_hz);
    let num_cpus = RaplCpu::device_count()?;
    let mut cpus = Vec::with_capacity(num_cpus);
    for cpu_id in 0..num_cpus {
        let cpu = RaplCpu::init(cpu_id)?;
        cpus.push((cpu_id, cpu));
    }
    Ok(start_cpu_poller(cpus, poll_hz))
}

#[cfg(not(target_os = "linux"))]
pub fn start_cpu_power_poller(_poll_hz: u32) -> anyhow::Result<CpuPowerPoller> {
    anyhow::bail!("CPU RAPL monitoring is only available on Linux.")
}

/// Reject API groups that aren't supported on this platform or that the
/// daemon lacks the privileges for.
#[allow(unused_variables)]
pub fn check_privileges(enabled_groups: &[ApiGroup]) -> anyhow::Result<()> {
    #[cfg(not(target_os = "linux"))]
    {
        if enabled_groups.contains(&ApiGroup::CpuRead) {
            tracing::error!(
                "API group 'cpu-read' is only supported on Linux \
                 (requires Intel RAPL via /sys/class/powercap)."
            );
            anyhow::bail!(
                "API group 'cpu-read' is only supported on Linux. \
                 Remove it from --enable to start zeusd on this platform."
            );
        }
    }

    #[cfg(unix)]
    {
        let is_root = nix::unistd::geteuid().is_root();
        for &group in enabled_groups {
            if group.requires_root() && !is_root {
                tracing::error!(
                    "API group '{}' requires root privileges. \
                     Either run as root or remove it from --enable.",
                    group,
                );
                anyhow::bail!(
                    "API group '{}' requires root but Zeusd is not running as root",
                    group,
                );
            }
        }
    }

    Ok(())
}

/// The set of enabled API groups, used as shared application state.
#[derive(Clone, Debug)]
pub struct EnabledGroups(pub HashSet<ApiGroup>);

/// Shared server state bundling all optional device handles and discovery info.
///
/// Fields are `Option` because devices are only initialized when their
/// corresponding API groups are enabled.
#[derive(Clone)]
pub struct ServerState {
    pub gpu_device_tasks: Option<GpuManagementTasks>,
    pub cpu_device_tasks: Option<CpuManagementTasks>,
    pub gpu_power_broadcast: Option<GpuPowerBroadcast>,
    pub cpu_power_broadcast: Option<CpuPowerBroadcast>,
    pub discovery_info: DiscoveryInfo,
    pub enabled_groups: EnabledGroups,
    pub signing_key: Option<SigningKeyData>,
}

/// Build an `App` with routes and data wired up from `$state`. Caller
/// clones `$state` if the App is built more than once.
macro_rules! build_app {
    ($state:expr) => {{
        let state = $state;
        let enabled = &state.enabled_groups.0;

        let mut app = App::new()
            .wrap(AuthMiddleware)
            .wrap(tracing_actix_web::TracingLogger::default())
            .configure(server_routes)
            .app_data(web::Data::new(state.discovery_info.clone()))
            .app_data(web::Data::new(state.enabled_groups.clone()));

        if let Some(ref key) = state.signing_key {
            app = app.app_data(web::Data::new(key.clone()));
        }

        if enabled.contains(&ApiGroup::GpuRead) || enabled.contains(&ApiGroup::GpuControl) {
            let mut gpu_scope = web::scope("/gpu");
            if enabled.contains(&ApiGroup::GpuRead) {
                gpu_scope = gpu_scope.configure(gpu_read_routes);
            }
            if enabled.contains(&ApiGroup::GpuControl) {
                gpu_scope = gpu_scope.configure(gpu_control_routes);
            }
            app = app.service(gpu_scope);
        }
        if let Some(ref tasks) = state.gpu_device_tasks {
            app = app.app_data(web::Data::new(tasks.clone()));
        }
        if let Some(ref broadcast) = state.gpu_power_broadcast {
            app = app.app_data(web::Data::new(broadcast.clone()));
        }

        if enabled.contains(&ApiGroup::CpuRead) {
            app = app.service(web::scope("/cpu").configure(cpu_routes));
        }
        if let Some(ref tasks) = state.cpu_device_tasks {
            app = app.app_data(web::Data::new(tasks.clone()));
        }
        if let Some(ref broadcast) = state.cpu_power_broadcast {
            app = app.app_data(web::Data::new(broadcast.clone()));
        }

        app
    }};
}

/// Build an `HttpServer` with routes and app data based on enabled API groups.
macro_rules! configure_server {
    ($state:expr, $workers:expr) => {
        HttpServer::new(move || build_app!($state.clone())).workers($workers)
    };
}

/// Set up routing and start the server on a unix domain socket.
#[cfg(unix)]
pub fn start_server_uds(
    listener: UnixListener,
    state: ServerState,
    num_workers: usize,
) -> std::io::Result<Server> {
    Ok(configure_server!(state, num_workers)
        .listen_uds(listener)?
        .run())
}

/// Set up routing and start the server over TCP.
pub fn start_server_tcp(
    listener: TcpListener,
    state: ServerState,
    num_workers: usize,
) -> std::io::Result<Server> {
    Ok(configure_server!(state, num_workers)
        .listen(listener)?
        .run())
}

/// Create a named pipe instance with the caller-supplied SDDL applied
/// to its DACL. The default elevated-process descriptor only grants
/// access to SYSTEM and Administrators, blocking unprivileged clients.
#[cfg(windows)]
fn create_pipe_with_sddl(
    pipe_name: &str,
    sddl: &str,
    first_instance: bool,
) -> std::io::Result<tokio::net::windows::named_pipe::NamedPipeServer> {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;
    use tokio::net::windows::named_pipe::ServerOptions;
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Authorization::{
        ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
    };
    use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;

    let sddl_w: Vec<u16> = OsStr::new(sddl)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let mut psd = std::ptr::null_mut();
    let ok = unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl_w.as_ptr(),
            SDDL_REVISION_1,
            &mut psd,
            std::ptr::null_mut(),
        )
    };
    if ok == 0 || psd.is_null() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "Failed to parse pipe SDDL `{}`: {}",
                sddl,
                std::io::Error::last_os_error()
            ),
        ));
    }

    let mut sa = SECURITY_ATTRIBUTES {
        nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: psd,
        bInheritHandle: 0,
    };

    let mut opts = ServerOptions::new();
    if first_instance {
        opts.first_pipe_instance(true);
    }
    opts.max_instances(254);

    // SAFETY: `sa` and `psd` are valid for the duration of this call. The
    // kernel makes its own copy of the security descriptor before the call
    // returns, so we can safely `LocalFree(psd)` immediately afterward.
    let result = unsafe {
        opts.create_with_security_attributes_raw(
            pipe_name,
            &mut sa as *mut _ as *mut std::ffi::c_void,
        )
    };
    unsafe { LocalFree(psd as _) };
    result
}

/// Serve actix-web over a Windows named pipe. `HttpServer` is Unix-only
/// there, so we drive `actix-http`'s `H1Service` directly on a single-
/// threaded `LocalSet` (per-connection futures are `!Send`).
#[cfg(windows)]
pub async fn run_server_named_pipe(
    pipe_name: String,
    pipe_sddl: String,
    state: ServerState,
) -> anyhow::Result<()> {
    use actix_http::HttpService;
    use actix_service::{IntoServiceFactory, Service, ServiceFactory, ServiceFactoryExt};
    use actix_web::dev::AppConfig;
    use std::rc::Rc;

    let initial = create_pipe_with_sddl(&pipe_name, &pipe_sddl, true)?;
    tracing::info!(
        "Listening on named pipe {} (SDDL: {})",
        &pipe_name,
        &pipe_sddl
    );
    let mut server: Option<tokio::net::windows::named_pipe::NamedPipeServer> = Some(initial);

    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            // Build inside the LocalSet (DateService ctor calls
            // `actix_rt::spawn`, which needs one) and ONCE for the daemon:
            // `AppInit::new_service` (actix-web 4.9 app_service.rs:77)
            // drains its services Vec on first call, so a second call
            // yields an empty router. Share the service via `Rc`.
            let app = build_app!(state);
            let svc_factory = HttpService::build().h1(actix_service::map_config(
                app.into_factory().map_err(|err| err.error_response()),
                |_| AppConfig::default(),
            ));
            let service = Rc::new(
                svc_factory
                    .new_service(())
                    .await
                    .map_err(|_| anyhow::anyhow!("Failed to initialize HTTP service"))?,
            );

            loop {
                let listener = match server.as_mut() {
                    Some(s) => s,
                    None => match create_pipe_with_sddl(&pipe_name, &pipe_sddl, false) {
                        Ok(new) => {
                            server = Some(new);
                            server.as_mut().unwrap()
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Cannot create named pipe listener: {e:?}; retrying in 1s. \
                                 This usually means the concurrent-connection cap was hit."
                            );
                            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                            continue;
                        }
                    },
                };

                if let Err(e) = listener.connect().await {
                    tracing::warn!("Named pipe accept failed: {e:?}; resetting listener");
                    server = None;
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    continue;
                }

                let connected = server.take().unwrap();
                server = match create_pipe_with_sddl(&pipe_name, &pipe_sddl, false) {
                    Ok(new) => Some(new),
                    Err(e) => {
                        tracing::warn!(
                            "Connected client OK but cannot create replacement listener: {e:?}; \
                             will retry on next loop iter"
                        );
                        None
                    }
                };

                let svc = Rc::clone(&service);
                tokio::task::spawn_local(async move {
                    if let Err(e) = svc.call((connected, None)).await {
                        tracing::warn!("Named pipe connection error: {e:?}");
                    }
                });
            }
            #[allow(unreachable_code)]
            Ok::<(), anyhow::Error>(())
        })
        .await
}
