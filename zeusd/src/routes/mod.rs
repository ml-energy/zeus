//! Routes and handlers for interacting with devices

pub mod cpu;
pub mod gpu;
pub mod server;

pub use cpu::{cpu_routes, CpuPowerSamplingPeriod};
pub use gpu::{gpu_control_routes, gpu_read_routes};
pub use server::{server_routes, CpuDiscoveryInfo, DiscoveryInfo, GpuDiscoveryInfo};

use actix_web::web::Bytes;
use actix_web::HttpResponse;
use futures::stream::{self, LocalBoxStream};
use futures::StreamExt;
use serde::Serialize;

use crate::power_streaming::PowerBroadcasts;

fn bad_request(error: String) -> HttpResponse {
    HttpResponse::BadRequest().json(serde_json::json!({ "error": error }))
}

/// Parse a comma-separated list of device indices, rejecting empty lists.
///
/// `device` is the device-type name used in error messages (e.g. `"CPU"`).
fn parse_device_ids(raw: &str, device: &str) -> Result<Vec<usize>, HttpResponse> {
    let param = format!("{}_ids", device.to_lowercase());
    if raw.trim().is_empty() {
        return Err(bad_request(format!(
            "{param} must contain at least one {device} index"
        )));
    }

    let mut ids = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            return Err(bad_request(format!(
                "{param} contains an empty {device} index"
            )));
        }
        let id = trimmed
            .parse()
            .map_err(|_| bad_request(format!("Invalid {device} index: {trimmed}")))?;
        ids.push(id);
    }
    ids.sort_unstable();
    ids.dedup();
    Ok(ids)
}

/// Resolve the optional device id list for a read endpoint: default to all
/// devices if absent, reject empty lists or out-of-range indices.
fn resolve_read_device_ids(
    query: &Option<String>,
    device_count: usize,
    device: &str,
) -> Result<Vec<usize>, HttpResponse> {
    match query {
        Some(raw) => {
            let parsed = parse_device_ids(raw, device)?;
            for &id in &parsed {
                if id >= device_count {
                    return Err(bad_request(format!("{device} {id} not found")));
                }
            }
            Ok(parsed)
        }
        None => Ok((0..device_count).collect()),
    }
}

/// Resolve the optional device id list for a stream endpoint against the
/// set of monitored devices, defaulting to all of them if absent.
fn resolve_stream_device_ids<T>(
    query: &Option<String>,
    broadcasts: &PowerBroadcasts<T>,
    device: &str,
) -> Result<Vec<usize>, HttpResponse>
where
    T: Clone + Default + Send + Sync + 'static,
{
    let ids = match query {
        Some(raw) => parse_device_ids(raw, device)?,
        None => broadcasts.valid_ids(),
    };
    if let Err(unknown) = broadcasts.validate_ids(&ids) {
        return Err(bad_request(format!(
            "Unknown {device} indices: {unknown:?}. Available: {:?}",
            broadcasts.valid_ids(),
        )));
    }
    Ok(ids)
}

/// Build an SSE response merging the requested devices' power sample streams.
///
/// Each subscriber guard keeps its poller active for the lifetime of the
/// stream.
fn power_stream_response<T>(device_ids: Vec<usize>, broadcasts: &PowerBroadcasts<T>) -> HttpResponse
where
    T: Clone + Default + Send + Sync + Serialize + 'static,
{
    let mut streams: Vec<LocalBoxStream<'static, Result<Bytes, actix_web::Error>>> =
        Vec::with_capacity(device_ids.len());
    for device_id in device_ids {
        let Some(broadcast) = broadcasts.get(device_id).cloned() else {
            continue;
        };
        let stream = broadcast.stream();
        let guard = broadcast.add_subscriber();
        streams.push(
            stream
                .map(move |sample| {
                    let _ = &guard;
                    let json = serde_json::to_string(&sample).unwrap_or_default();
                    Ok::<_, actix_web::Error>(Bytes::from(format!("data: {json}\n\n")))
                })
                .boxed_local(),
        );
    }
    let stream: LocalBoxStream<'static, Result<Bytes, actix_web::Error>> = if streams.is_empty() {
        stream::pending().boxed_local()
    } else {
        stream::select_all(streams).boxed_local()
    };
    HttpResponse::Ok()
        .insert_header(("Content-Type", "text/event-stream"))
        .insert_header(("Cache-Control", "no-cache"))
        .streaming(stream)
}
