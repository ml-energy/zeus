//! Routes for interacting with CPUs

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use actix_web::{web, HttpResponse};
use serde::{Deserialize, Serialize};
use tokio::time::{sleep, Duration};

use super::{power_stream_response, resolve_read_device_ids, resolve_stream_device_ids};
use crate::devices::cpu::power::{CpuDramPower, CpuPowerBroadcasts, CpuPowerSnapshot};
use crate::devices::cpu::{CpuCommand, CpuManagementTasks, RaplResponse};
use crate::error::{aggregate_error_response, ZeusdError};
use crate::power_streaming::unix_timestamp_ms;

/// Query parameters for CPU read endpoints.
/// `cpu_ids` is optional; omit to read all CPUs.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpuReadQuery {
    pub cpu_ids: Option<String>,
}

#[derive(Clone, Copy)]
pub struct CpuPowerSamplingPeriod {
    period_us: u64,
}

impl CpuPowerSamplingPeriod {
    pub fn from_poll_hz(poll_hz: u32) -> Self {
        Self {
            period_us: 1_000_000u64 / poll_hz.max(1) as u64,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct GetCumulativeEnergy {
    pub cpu_ids: Option<String>,
    pub cpu: bool,
    pub dram: bool,
}

impl From<GetCumulativeEnergy> for CpuCommand {
    fn from(request: GetCumulativeEnergy) -> Self {
        CpuCommand::GetIndexEnergy {
            cpu: request.cpu,
            dram: request.dram,
        }
    }
}

#[actix_web::get("/get_cumulative_energy")]
#[tracing::instrument(
    skip(query, device_tasks),
    fields(
        cpu_ids = ?query.cpu_ids,
        cpu = %query.cpu,
        dram = %query.dram,
    )
)]
async fn get_cumulative_energy_handler(
    query: web::Query<GetCumulativeEnergy>,
    device_tasks: web::Data<CpuManagementTasks>,
) -> Result<HttpResponse, ZeusdError> {
    let now = Instant::now();

    let cpu_ids = match resolve_read_device_ids(&query.cpu_ids, device_tasks.device_count(), "CPU")
    {
        Ok(ids) => ids,
        Err(resp) => return Ok(resp),
    };

    // Execute concurrently for all requested CPUs.
    let mut handles = Vec::with_capacity(cpu_ids.len());
    for &cpu_id in &cpu_ids {
        let cmd: CpuCommand = CpuCommand::GetIndexEnergy {
            cpu: query.cpu,
            dram: query.dram,
        };
        let tasks = device_tasks.clone();
        handles.push(async move { (cpu_id, tasks.send_command_blocking(cpu_id, cmd, now).await) });
    }
    let results = futures::future::join_all(handles).await;

    let mut response_map: HashMap<String, RaplResponse> = HashMap::new();
    let mut errors: HashMap<usize, ZeusdError> = HashMap::new();
    for (cpu_id, result) in results {
        match result {
            Ok(measurement) => {
                response_map.insert(cpu_id.to_string(), measurement);
            }
            Err(e) => {
                errors.insert(cpu_id, e);
            }
        }
    }

    if errors.is_empty() {
        Ok(HttpResponse::Ok().json(response_map))
    } else {
        Ok(aggregate_error_response(errors))
    }
}

async fn read_cpu_energy_for_power(
    cpu_ids: &[usize],
    device_tasks: &CpuManagementTasks,
) -> (HashMap<usize, RaplResponse>, HashMap<usize, ZeusdError>) {
    let now = Instant::now();
    let mut handles = Vec::with_capacity(cpu_ids.len());
    for &cpu_id in cpu_ids {
        let tasks = device_tasks.clone();
        handles.push(async move {
            (
                cpu_id,
                tasks
                    .send_command_blocking(
                        cpu_id,
                        CpuCommand::GetIndexEnergy {
                            cpu: true,
                            dram: true,
                        },
                        now,
                    )
                    .await,
            )
        });
    }

    let results = futures::future::join_all(handles).await;
    let mut responses = HashMap::new();
    let mut errors = HashMap::new();
    for (cpu_id, result) in results {
        match result {
            Ok(response) => {
                responses.insert(cpu_id, response);
            }
            Err(e) => {
                errors.insert(cpu_id, e);
            }
        }
    }
    (responses, errors)
}

fn compute_cpu_power(
    cpu_id: usize,
    first: &RaplResponse,
    second: &RaplResponse,
    elapsed_us: u64,
) -> Result<CpuDramPower, ZeusdError> {
    let first_cpu = first
        .cpu_energy_uj
        .ok_or(ZeusdError::CpuPowerMeasurementError(cpu_id))?;
    let second_cpu = second
        .cpu_energy_uj
        .ok_or(ZeusdError::CpuPowerMeasurementError(cpu_id))?;
    let cpu_mw = (second_cpu.saturating_sub(first_cpu) * 1000 / elapsed_us) as u32;

    let dram_mw = match (first.dram_energy_uj, second.dram_energy_uj) {
        (Some(first_dram), Some(second_dram)) => {
            Some((second_dram.saturating_sub(first_dram) * 1000 / elapsed_us) as u32)
        }
        (None, None) => None,
        _ => return Err(ZeusdError::CpuPowerMeasurementError(cpu_id)),
    };

    Ok(CpuDramPower { cpu_mw, dram_mw })
}

fn cpu_power_snapshot(
    cpu_ids: &[usize],
    first: &HashMap<usize, RaplResponse>,
    second: &HashMap<usize, RaplResponse>,
    elapsed_us: u64,
) -> Result<CpuPowerSnapshot, HashMap<usize, ZeusdError>> {
    let mut power_mw = BTreeMap::new();
    let mut errors = HashMap::new();

    for &cpu_id in cpu_ids {
        match (first.get(&cpu_id), second.get(&cpu_id)) {
            (Some(first), Some(second)) => {
                match compute_cpu_power(cpu_id, first, second, elapsed_us) {
                    Ok(power) => {
                        power_mw.insert(cpu_id, power);
                    }
                    Err(e) => {
                        errors.insert(cpu_id, e);
                    }
                }
            }
            _ => {
                errors.insert(cpu_id, ZeusdError::CpuPowerMeasurementError(cpu_id));
            }
        }
    }

    if errors.is_empty() {
        Ok(CpuPowerSnapshot {
            timestamp_ms: unix_timestamp_ms(),
            power_mw,
        })
    } else {
        Err(errors)
    }
}

/// One-shot CPU power reading (computed from RAPL energy deltas).
///
/// Reads only the requested CPUs twice, separated by the configured power
/// sampling period, and computes power over the measured elapsed time.
#[actix_web::get("/get_power")]
#[tracing::instrument(skip(device_tasks, period), fields(cpu_ids = ?query.cpu_ids))]
async fn get_cpu_power_handler(
    query: web::Query<CpuReadQuery>,
    device_tasks: web::Data<CpuManagementTasks>,
    period: web::Data<CpuPowerSamplingPeriod>,
) -> HttpResponse {
    let cpu_ids = match resolve_read_device_ids(&query.cpu_ids, device_tasks.device_count(), "CPU")
    {
        Ok(ids) => ids,
        Err(resp) => return resp,
    };

    let (first, mut errors) = read_cpu_energy_for_power(&cpu_ids, device_tasks.get_ref()).await;
    if !errors.is_empty() {
        return aggregate_error_response(errors);
    }

    let first_read_done = Instant::now();
    sleep(Duration::from_micros(period.period_us)).await;

    let (second, second_errors) = read_cpu_energy_for_power(&cpu_ids, device_tasks.get_ref()).await;
    let elapsed_us = (first_read_done.elapsed().as_micros() as u64).max(1);
    errors.extend(second_errors);
    if !errors.is_empty() {
        return aggregate_error_response(errors);
    }

    match cpu_power_snapshot(&cpu_ids, &first, &second, elapsed_us) {
        Ok(snapshot) => HttpResponse::Ok().json(snapshot),
        Err(errors) => aggregate_error_response(errors),
    }
}

/// SSE stream of CPU power readings.
///
/// The subscriber guard keeps the poller active for the lifetime of the stream.
#[actix_web::get("/stream_power")]
#[tracing::instrument(skip(broadcast), fields(cpu_ids = ?query.cpu_ids))]
async fn cpu_power_stream_handler(
    query: web::Query<CpuReadQuery>,
    broadcast: web::Data<CpuPowerBroadcasts>,
) -> HttpResponse {
    match resolve_stream_device_ids(&query.cpu_ids, broadcast.get_ref(), "CPU") {
        Ok(cpu_ids) => power_stream_response(cpu_ids, broadcast.get_ref()),
        Err(response) => response,
    }
}

pub fn cpu_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(get_cumulative_energy_handler)
        .service(get_cpu_power_handler)
        .service(cpu_power_stream_handler);
}
