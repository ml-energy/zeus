//! CPU power measurement with RAPL. Only supported on Linux.

use std::io::Read;

use tokio::io::AsyncReadExt;

pub struct FieldInfo {
    pub name: String,
    pub max_energy_uj: f64,
}

pub trait CpuManager {
    fn get_available_fields(&self) -> Vec<FieldInfo>;
    fn get_field_energy(&self, field_name: &str) -> f64;
}

static RAPL_DIR: &'static str = "/sys/class/powercap/intel-rapl";

pub struct RAPLCpuManager {
    fields: Vec<FieldInfo>,
}

impl RAPLCpuManager {
    pub fn new() -> Self {
        let mut fields = Vec::with_capacity(2);

        // The package domain is always present

        // Look for subdomains and add to fields

        Self { fields }
    }
}

fn read_u64(path: &std::path::Path) -> f64 {
    let mut file = std::fs::File::open(path).unwrap();
    let mut buf = String::new();
    file.read_to_string(&mut buf).unwrap();
    buf.trim().parse().unwrap()
}

async fn read_u64_async(path: &std::path::Path) -> f64 {
    let mut file = tokio::fs::File::open(path).await.unwrap();
    let mut buf = String::new();
    file.read_to_string(&mut buf).await.unwrap();
    buf.trim().parse().unwrap()
}
