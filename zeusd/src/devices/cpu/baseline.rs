//! Baseline CPU power limit settings that `reset_power_limit` restores.
//!
//! Neither RAPL nor HSMP reports a default power limit, so zeusd records the
//! settings it finds when it first starts after boot. The baseline file
//! defaults to a path under `/run`, which is cleared on reboot. The recorded
//! values are then the BIOS settings, unless something changed them before
//! zeusd first started.

use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Context;
use serde::{Deserialize, Serialize};

/// Power limit and time window of one package zone constraint.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ConstraintSetting {
    pub name: String,
    /// Power limit in microwatts, the unit of the kernel's powercap interface.
    pub power_limit_uw: u64,
    /// Time window in microseconds, or None if the constraint has none.
    pub time_window_us: Option<u64>,
}

/// Recorded constraint settings of one CPU package zone.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PackageBaseline {
    /// Name of the package zone, such as `package-0`.
    pub zone: String,
    pub constraints: Vec<ConstraintSetting>,
}

/// On-disk format of the baseline file. `cpus` is indexed by CPU ID.
#[derive(Serialize, Deserialize, Debug)]
struct BaselineFile {
    cpus: Vec<PackageBaseline>,
}

/// Load the baseline at `path`, or record `current` there if the file does not exist.
///
/// Errors if the loaded baseline does not have the same package zones and
/// constraint names as `current`.
pub fn load_or_record(
    path: &Path,
    current: Vec<PackageBaseline>,
) -> anyhow::Result<Vec<PackageBaseline>> {
    if record(path, &current)? {
        tracing::info!(
            "Recorded the current CPU power limit settings as the baseline at {}",
            path.display()
        );
        return Ok(current);
    }

    let contents = fs::read_to_string(path).with_context(|| {
        format!(
            "Failed to read the CPU power limit baseline at {}",
            path.display()
        )
    })?;
    let file: BaselineFile = serde_json::from_str(&contents).with_context(|| {
        format!(
            "Failed to parse the CPU power limit baseline at {}. {DELETE_HINT}",
            path.display()
        )
    })?;
    check_matches(&file.cpus, &current).with_context(|| {
        format!(
            "The CPU power limit baseline at {} does not match this machine. {DELETE_HINT}",
            path.display()
        )
    })?;
    tracing::info!("Loaded CPU power limit baseline from {}", path.display());
    Ok(file.cpus)
}

const DELETE_HINT: &str = "Reboot, or restore the power limits to the values you want \
    reset to restore and delete the file, to record a new baseline.";

/// Publish the baseline at `path` unless one already exists there, and
/// return whether this call published it.
///
/// The contents go to a temporary file that is then hard-linked to `path`.
/// Linking fails if `path` exists, so concurrent starts never replace a
/// published baseline or expose a partially written one.
fn record(path: &Path, cpus: &[PackageBaseline]) -> anyhow::Result<bool> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory {}", parent.display()))?;
        }
    }
    let contents = serde_json::to_string_pretty(&BaselineFile {
        cpus: cpus.to_vec(),
    })?;
    let mut tmp_name = path.as_os_str().to_owned();
    static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(0);
    tmp_name.push(format!(
        ".tmp.{}.{}",
        std::process::id(),
        NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let tmp_path = PathBuf::from(tmp_name);
    fs::write(&tmp_path, contents)
        .with_context(|| format!("Failed to write {}", tmp_path.display()))?;
    let linked = fs::hard_link(&tmp_path, path);
    fs::remove_file(&tmp_path)
        .with_context(|| format!("Failed to remove {}", tmp_path.display()))?;
    match linked {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == ErrorKind::AlreadyExists => Ok(false),
        Err(e) => Err(e).with_context(|| format!("Failed to publish {}", path.display())),
    }
}

fn check_matches(recorded: &[PackageBaseline], current: &[PackageBaseline]) -> anyhow::Result<()> {
    if recorded.len() != current.len() {
        anyhow::bail!(
            "it has {} CPU package(s), but the machine has {}",
            recorded.len(),
            current.len()
        );
    }
    for (cpu_id, (recorded, current)) in recorded.iter().zip(current).enumerate() {
        if recorded.zone != current.zone {
            anyhow::bail!(
                "CPU {cpu_id} is zone '{}' in it, but '{}' on the machine",
                recorded.zone,
                current.zone
            );
        }
        let names = |package: &PackageBaseline| -> Vec<String> {
            package.constraints.iter().map(|c| c.name.clone()).collect()
        };
        if names(recorded) != names(current) {
            anyhow::bail!(
                "CPU {cpu_id} has constraints {:?} in it, but {:?} on the machine",
                names(recorded),
                names(current)
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn package(zone: &str, settings: &[(&str, u64, Option<u64>)]) -> PackageBaseline {
        PackageBaseline {
            zone: zone.to_string(),
            constraints: settings
                .iter()
                .map(
                    |&(name, power_limit_uw, time_window_us)| ConstraintSetting {
                        name: name.to_string(),
                        power_limit_uw,
                        time_window_us,
                    },
                )
                .collect(),
        }
    }

    #[test]
    fn records_when_missing_and_loads_afterwards() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("zeusd").join("baseline.json");
        let boot = vec![package(
            "package-0",
            &[
                ("long_term", 205_000_000, Some(999_424)),
                ("short_term", 246_000_000, Some(2_440)),
            ],
        )];

        assert_eq!(load_or_record(&path, boot.clone()).unwrap(), boot);
        assert!(path.exists());
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);

        // A later start sees changed limits but keeps the recorded baseline.
        let changed = vec![package(
            "package-0",
            &[
                ("long_term", 150_000_000, Some(999_424)),
                ("short_term", 246_000_000, Some(2_440)),
            ],
        )];
        assert_eq!(load_or_record(&path, changed).unwrap(), boot);
    }

    #[test]
    fn mismatched_baseline_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("baseline.json");
        let boot = vec![package("package-0", &[("socket", 200_000_000, None)])];
        load_or_record(&path, boot).unwrap();

        let more_packages = vec![
            package("package-0", &[("socket", 200_000_000, None)]),
            package("package-1", &[("socket", 200_000_000, None)]),
        ];
        assert!(load_or_record(&path, more_packages).is_err());

        let other_zone = vec![package("package-1", &[("socket", 200_000_000, None)])];
        assert!(load_or_record(&path, other_zone).is_err());

        let other_constraints = vec![package("package-0", &[])];
        assert!(load_or_record(&path, other_constraints).is_err());
    }

    #[test]
    fn corrupt_baseline_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("baseline.json");
        fs::write(&path, "{").unwrap();
        assert!(load_or_record(&path, vec![]).is_err());
    }

    #[test]
    fn concurrent_starts_agree_on_one_baseline() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("baseline.json");

        let results: Vec<Vec<PackageBaseline>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..16u64)
                .map(|i| {
                    let path = &path;
                    scope.spawn(move || {
                        let current =
                            vec![package("package-0", &[("socket", 100_000_000 + i, None)])];
                        load_or_record(path, current).unwrap()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        assert!(results.iter().all(|r| *r == results[0]));
        assert_eq!(
            load_or_record(&path, results[0].clone()).unwrap(),
            results[0]
        );
        assert_eq!(fs::read_dir(tmp.path()).unwrap().count(), 1);
    }
}
