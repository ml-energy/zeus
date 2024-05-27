//! Helpers for running integration tests.
//!
//! It has to be under `tests/helpers/mod.rs` instead of `tests/helpers.rs`
//! to avoid it from being treated as another test module.

use nvml_wrapper::error::NvmlError;
use once_cell::sync::Lazy;
use paste::paste;
use std::future::Future;
use std::net::TcpListener;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use zeusd::devices::gpu::{GpuManagementTasks, GpuManager};
use zeusd::error::ZeusdError;
use zeusd::startup::{init_tracing, start_server_tcp};

static NUM_GPUS: u32 = 4;

static TRACING: Lazy<()> = Lazy::new(|| {
    if std::env::var("TEST_LOG").is_ok() {
        init_tracing(std::io::stdout).expect("Failed to initialize tracing");
    } else {
        init_tracing(std::io::sink).expect("Failed to initialize tracing");
    };
});

#[derive(Clone)]
pub struct TestGpu {
    persistent_mode_tx: UnboundedSender<bool>,
    power_limit_tx: UnboundedSender<u32>,
    gpu_locked_clocks_tx: UnboundedSender<(u32, u32)>,
    mem_locked_clocks_tx: UnboundedSender<(u32, u32)>,
    valid_power_limit_range: (u32, u32),
}

pub struct TestGpuObserver {
    persistent_mode_rx: UnboundedReceiver<bool>,
    power_limit_rx: UnboundedReceiver<u32>,
    gpu_locked_clocks_rx: UnboundedReceiver<(u32, u32)>,
    mem_locked_clocks_rx: UnboundedReceiver<(u32, u32)>,
}

impl TestGpu {
    fn init() -> Result<(Self, TestGpuObserver), ZeusdError> {
        let (persistent_mode_tx, persistent_mode_rx) = tokio::sync::mpsc::unbounded_channel();
        let (power_limit_tx, power_limit_rx) = tokio::sync::mpsc::unbounded_channel();
        let (gpu_locked_clocks_tx, gpu_locked_clocks_rx) = tokio::sync::mpsc::unbounded_channel();
        let (mem_locked_clocks_tx, mem_locked_clocks_rx) = tokio::sync::mpsc::unbounded_channel();

        let gpu = TestGpu {
            persistent_mode_tx,
            power_limit_tx,
            gpu_locked_clocks_tx,
            mem_locked_clocks_tx,
            valid_power_limit_range: (100_000, 300_000),
        };
        let observer = TestGpuObserver {
            persistent_mode_rx,
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

    fn set_persistent_mode(&mut self, enabled: bool) -> Result<(), ZeusdError> {
        self.persistent_mode_tx.send(enabled).unwrap();
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

pub fn start_test_tasks() -> anyhow::Result<(GpuManagementTasks, Vec<TestGpuObserver>)> {
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

/// A helper trait for building URLs to send requests to.
pub trait ZeusdRequest: serde::Serialize {
    fn build_url(app: &TestApp, gpu_id: u32) -> String;
}

macro_rules! impl_zeusd_request {
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

impl_zeusd_request!(SetPersistentMode);
impl_zeusd_request!(SetPowerLimit);
impl_zeusd_request!(SetGpuLockedClocks);
impl_zeusd_request!(ResetGpuLockedClocks);
impl_zeusd_request!(SetMemLockedClocks);
impl_zeusd_request!(ResetMemLockedClocks);

/// A test application that starts a server over TCP and provides helper methods
/// for sending requests and fetching what happened to the fake GPUs.
pub struct TestApp {
    port: u16,
    observers: Vec<TestGpuObserver>,
}

impl TestApp {
    pub async fn start() -> Self {
        Lazy::force(&TRACING);

        let (test_tasks, test_gpu_observers) =
            start_test_tasks().expect("Failed to start test tasks");

        let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind TCP listener");
        let port = listener.local_addr().unwrap().port();
        let server = start_server_tcp(listener, test_tasks, 8).expect("Failed to start server");
        let _ = tokio::spawn(async move { server.await });

        TestApp {
            port,
            observers: test_gpu_observers,
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

    pub fn persistent_mode_history_for_gpu(&mut self, gpu_id: usize) -> Vec<bool> {
        let rx = &mut self.observers[gpu_id].persistent_mode_rx;
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
}
