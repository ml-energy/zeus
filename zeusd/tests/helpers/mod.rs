use once_cell::sync::Lazy;
use std::net::TcpListener;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use zeusd::devices::gpu::{GpuManagementTasks, GpuManager};
use zeusd::error::ZeusdError;
use zeusd::routes::gpu::SetPersistentModeRequest;
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
    pub persistent_mode_tx: UnboundedSender<bool>,
    pub power_limit_tx: UnboundedSender<u32>,
    pub gpu_locked_clocks_tx: UnboundedSender<(u32, u32)>,
    pub mem_locked_clocks_tx: UnboundedSender<(u32, u32)>,
}

pub struct TestGpuObserver {
    pub persistent_mode_rx: UnboundedReceiver<bool>,
    pub power_limit_rx: UnboundedReceiver<u32>,
    pub gpu_locked_clocks_rx: UnboundedReceiver<(u32, u32)>,
    pub mem_locked_clocks_rx: UnboundedReceiver<(u32, u32)>,
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

pub struct TestApp {
    port: u16,
    pub observers: Vec<TestGpuObserver>,
}

impl TestApp {
    pub async fn start() -> Self {
        Lazy::force(&TRACING);

        let (test_tasks, test_gpu_observers) =
            start_test_tasks().expect("Failed to start test tasks");

        let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind TCP listener");
        let port = listener.local_addr().unwrap().port();
        let server = start_server_tcp(listener, test_tasks).expect("Failed to start server");
        let _ = tokio::spawn(async move { server.await });

        TestApp {
            port,
            observers: test_gpu_observers,
        }
    }

    pub async fn set_persistent_mode(
        &mut self,
        gpu_id: u32,
        enabled: bool,
        block: bool,
    ) -> reqwest::Response {
        let client = reqwest::Client::new();
        let url = format!(
            "http://127.0.0.1:{}/gpu/{}/persistent_mode",
            self.port, gpu_id
        );
        let payload = SetPersistentModeRequest { enabled, block };

        client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .expect("Failed to send request")
    }
}
