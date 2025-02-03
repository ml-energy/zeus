//! Helpers for running integration tests.
//!
//! It has to be under `tests/helpers/mod.rs` instead of `tests/helpers.rs`
//! to avoid it from being treated as another test module.

use nvml_wrapper::error::NvmlError;
use once_cell::sync::Lazy;
use paste::paste;
use std::future::Future;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use zeusd::devices::cpu::{CpuManagementTasks, CpuManager, PackageInfo};
use zeusd::devices::gpu::{GpuManagementTasks, GpuManager};
use zeusd::error::ZeusdError;
use zeusd::startup::{init_tracing, start_server_tcp};

static NUM_GPUS: u32 = 4;

static NUM_CPUS: usize = 1;

static TRACING: Lazy<()> = Lazy::new(|| {
    if std::env::var("TEST_LOG").is_ok() {
        init_tracing(std::io::stdout).expect("Failed to initialize tracing");
    } else {
        init_tracing(std::io::sink).expect("Failed to initialize tracing");
    };
});

#[derive(Clone)]
pub struct TestGpu {
    persistence_mode_tx: UnboundedSender<bool>,
    power_limit_tx: UnboundedSender<u32>,
    gpu_locked_clocks_tx: UnboundedSender<(u32, u32)>,
    mem_locked_clocks_tx: UnboundedSender<(u32, u32)>,
    valid_power_limit_range: (u32, u32),
}

pub struct TestGpuObserver {
    persistence_mode_rx: UnboundedReceiver<bool>,
    power_limit_rx: UnboundedReceiver<u32>,
    gpu_locked_clocks_rx: UnboundedReceiver<(u32, u32)>,
    mem_locked_clocks_rx: UnboundedReceiver<(u32, u32)>,
}

impl TestGpu {
    fn init() -> Result<(Self, TestGpuObserver), ZeusdError> {
        let (persistence_mode_tx, persistence_mode_rx) = tokio::sync::mpsc::unbounded_channel();
        let (power_limit_tx, power_limit_rx) = tokio::sync::mpsc::unbounded_channel();
        let (gpu_locked_clocks_tx, gpu_locked_clocks_rx) = tokio::sync::mpsc::unbounded_channel();
        let (mem_locked_clocks_tx, mem_locked_clocks_rx) = tokio::sync::mpsc::unbounded_channel();

        let gpu = TestGpu {
            persistence_mode_tx,
            power_limit_tx,
            gpu_locked_clocks_tx,
            mem_locked_clocks_tx,
            valid_power_limit_range: (100_000, 300_000),
        };
        let observer = TestGpuObserver {
            persistence_mode_rx,
            power_limit_rx,
            gpu_locked_clocks_rx,
            mem_locked_clocks_rx,
        };

        Ok((gpu, observer))
    }
}

impl GpuManager for TestGpu {
    fn device_count() -> Result<u32, ZeusdError> {
        Ok(NUM_GPUS)
    }

    fn set_persistence_mode(&mut self, enabled: bool) -> Result<(), ZeusdError> {
        self.persistence_mode_tx.send(enabled).unwrap();
        Ok(())
    }

    fn set_power_management_limit(&mut self, power_limit: u32) -> Result<(), ZeusdError> {
        if power_limit < self.valid_power_limit_range.0
            || power_limit > self.valid_power_limit_range.1
        {
            return Err(ZeusdError::from(NvmlError::InvalidArg));
        }
        self.power_limit_tx.send(power_limit).unwrap();
        Ok(())
    }

    fn set_gpu_locked_clocks(
        &mut self,
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    ) -> Result<(), ZeusdError> {
        self.gpu_locked_clocks_tx
            .send((min_clock_mhz, max_clock_mhz))
            .unwrap();
        Ok(())
    }

    fn reset_gpu_locked_clocks(&mut self) -> Result<(), ZeusdError> {
        self.gpu_locked_clocks_tx.send((0, 0)).unwrap();
        Ok(())
    }

    fn set_mem_locked_clocks(
        &mut self,
        min_clock_mhz: u32,
        max_clock_mhz: u32,
    ) -> Result<(), ZeusdError> {
        self.mem_locked_clocks_tx
            .send((min_clock_mhz, max_clock_mhz))
            .unwrap();
        Ok(())
    }

    fn reset_mem_locked_clocks(&mut self) -> Result<(), ZeusdError> {
        self.mem_locked_clocks_tx.send((0, 0)).unwrap();
        Ok(())
    }
}

pub struct TestCpu {
    pub cpu: UnboundedReceiver<u64>,
    pub dram: UnboundedReceiver<u64>,
}

pub struct TestCpuInjector {
    pub cpu: UnboundedSender<u64>,
    pub dram: UnboundedSender<u64>,
}

impl TestCpu {
    fn init(_index: usize) -> Result<(Self, TestCpuInjector), ZeusdError> {
        let (cpu_sender, cpu_receiver) = tokio::sync::mpsc::unbounded_channel();
        let (dram_sender, dram_receiver) = tokio::sync::mpsc::unbounded_channel();
        Ok((
            TestCpu {
                cpu: cpu_receiver,
                dram: dram_receiver,
            },
            TestCpuInjector {
                cpu: cpu_sender,
                dram: dram_sender,
            },
        ))
    }
}

impl CpuManager for TestCpu {
    fn device_count() -> Result<usize, ZeusdError> {
        Ok(1)
    }

    fn get_available_fields(
        _index: usize,
    ) -> Result<(Arc<PackageInfo>, Option<Arc<PackageInfo>>), ZeusdError> {
        Ok((
            Arc::new(PackageInfo {
                index: _index,
                name: "package-0".to_string(),
                energy_uj_path: PathBuf::from(
                    "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
                ),
                max_energy_uj: 1000000,
                num_wraparounds: RwLock::new(0),
            }),
            Some(Arc::new(PackageInfo {
                index: _index,
                name: "dram".to_string(),
                energy_uj_path: PathBuf::from(
                    "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj",
                ),
                max_energy_uj: 1000000,
                num_wraparounds: RwLock::new(0),
            })),
        ))
    }

    fn get_cpu_energy(&mut self) -> Result<u64, ZeusdError> {
        Ok(self.cpu.try_recv().ok().unwrap())
    }

    fn get_dram_energy(&mut self) -> Result<u64, ZeusdError> {
        Ok(self.dram.try_recv().ok().unwrap())
    }

    fn stop_monitoring(&mut self) {}

    fn is_dram_available(&self) -> bool {
        true
    }
}

pub fn start_gpu_test_tasks() -> anyhow::Result<(GpuManagementTasks, Vec<TestGpuObserver>)> {
    let mut gpus = Vec::with_capacity(4);
    let mut observers = Vec::with_capacity(4);
    for _ in 0..4 {
        let (gpu, observer) = TestGpu::init()?;
        gpus.push(gpu);
        observers.push(observer);
    }

    let tasks = GpuManagementTasks::start(gpus)?;

    Ok((tasks, observers))
}

pub fn start_cpu_test_tasks() -> anyhow::Result<(CpuManagementTasks, Vec<TestCpuInjector>)> {
    let mut cpus = Vec::with_capacity(NUM_CPUS);
    let mut injectors = Vec::with_capacity(NUM_CPUS);
    for i in 0..NUM_CPUS {
        let (cpu, cpu_injector) = TestCpu::init(i)?;
        cpus.push(cpu);
        injectors.push(cpu_injector)
    }
    let tasks = CpuManagementTasks::start(cpus)?;
    Ok((tasks, injectors))
}

/// A helper trait for building URLs to send requests to.
pub trait ZeusdRequest: serde::Serialize {
    fn build_url(app: &TestApp, gpu_id: u32) -> String;
}

macro_rules! impl_zeusd_request_gpu {
    ($api:ident) => {
        paste! {
            impl ZeusdRequest for zeusd::routes::gpu::[<$api:camel>] {
                fn build_url(app: &TestApp, gpu_id: u32) -> String {
                    format!(
                        "http://127.0.0.1:{}/gpu/{}/{}",
                        app.port, gpu_id, stringify!([<$api:snake>]),
                    )
                }
            }
        }
    };
}

macro_rules! impl_zeusd_request_cpu {
    ($api:ident) => {
        paste! {
            impl ZeusdRequest for zeusd::routes::cpu::[<$api:camel>] {
                fn build_url(app: &TestApp, cpu_id: u32) -> String {
                    format!(
                        "http://127.0.0.1:{}/cpu/{}/{}",
                        app.port, cpu_id, stringify!([<$api:snake>]),
                    )
                }
            }
        }
    };
}
impl_zeusd_request_gpu!(SetPersistenceMode);
impl_zeusd_request_gpu!(SetPowerLimit);
impl_zeusd_request_gpu!(SetGpuLockedClocks);
impl_zeusd_request_gpu!(ResetGpuLockedClocks);
impl_zeusd_request_gpu!(SetMemLockedClocks);
impl_zeusd_request_gpu!(ResetMemLockedClocks);

impl_zeusd_request_cpu!(GetIndexEnergy);

/// A test application that starts a server over TCP and provides helper methods
/// for sending requests and fetching what happened to the fake GPUs.
pub struct TestApp {
    pub port: u16,
    observers: Vec<TestGpuObserver>,
    cpu_injectors: Vec<TestCpuInjector>,
}

impl TestApp {
    pub async fn start() -> Self {
        Lazy::force(&TRACING);

        let (gpu_test_tasks, test_gpu_observers) =
            start_gpu_test_tasks().expect("Failed to start gpu test tasks");

        let (cpu_test_tasks, cpu_test_injectors) =
            start_cpu_test_tasks().expect("Failed to start cpu test tasks");

        let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind TCP listener");
        let port = listener.local_addr().unwrap().port();
        let server = start_server_tcp(listener, gpu_test_tasks, cpu_test_tasks, 2)
            .expect("Failed to start server");
        let _ = tokio::spawn(async move { server.await });

        TestApp {
            port,
            observers: test_gpu_observers,
            cpu_injectors: cpu_test_injectors,
        }
    }

    pub fn send<T: ZeusdRequest>(
        &mut self,
        gpu_id: u32,
        payload: T,
    ) -> impl Future<Output = Result<reqwest::Response, reqwest::Error>> {
        let client = reqwest::Client::new();
        let url = T::build_url(self, gpu_id);

        client.post(url).json(&payload).send()
    }

    pub fn persistence_mode_history_for_gpu(&mut self, gpu_id: usize) -> Vec<bool> {
        let rx = &mut self.observers[gpu_id].persistence_mode_rx;
        std::iter::from_fn(|| rx.try_recv().ok()).collect()
    }

    pub fn power_limit_history_for_gpu(&mut self, gpu_id: usize) -> Vec<u32> {
        let rx = &mut self.observers[gpu_id].power_limit_rx;
        std::iter::from_fn(|| rx.try_recv().ok()).collect()
    }

    pub fn gpu_locked_clocks_history_for_gpu(&mut self, gpu_id: usize) -> Vec<(u32, u32)> {
        let rx = &mut self.observers[gpu_id].gpu_locked_clocks_rx;
        std::iter::from_fn(|| rx.try_recv().ok()).collect()
    }

    pub fn mem_locked_clocks_history_for_gpu(&mut self, gpu_id: usize) -> Vec<(u32, u32)> {
        let rx = &mut self.observers[gpu_id].mem_locked_clocks_rx;
        std::iter::from_fn(|| rx.try_recv().ok()).collect()
    }

    pub fn set_cpu_energy_measurements(&mut self, cpu_id: usize, measurements: &Vec<u64>) {
        for measurement in measurements {
            self.cpu_injectors[cpu_id].cpu.send(*measurement).unwrap();
        }
    }

    pub fn set_dram_energy_measurements(&mut self, cpu_id: usize, measurements: &Vec<u64>) {
        for measurement in measurements {
            self.cpu_injectors[cpu_id].dram.send(*measurement).unwrap();
        }
    }
}
