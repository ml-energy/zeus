//! Original CPU power limit settings that `reset_power_limit` restores.
//!
//! Neither RAPL nor HSMP reports a default power limit, so the original
//! settings are the ones zeusd finds when it starts. With persistent storage,
//! the file name of the original snapshot carries the kernel's boot ID. The
//! first zeusd start in a host boot records the settings it finds, and later
//! starts in the same boot, including replacement containers that mount the
//! same directory, load them.

use std::ffi::OsString;
use std::fs;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::ORIGINAL_POWER_LIMITS_DOC_URL;

/// File in which the kernel reports a random ID that changes on every boot.
pub const BOOT_ID_PATH: &str = "/proc/sys/kernel/random/boot_id";

/// Power limit and time window of one package zone constraint.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ConstraintSetting {
    pub name: String,
    /// Power limit in microwatts, the unit of the kernel's powercap interface.
    pub power_limit_uw: u64,
    /// Time window in microseconds, or None if the constraint has none.
    pub time_window_us: Option<u64>,
}

/// Constraint settings of one CPU package zone.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PackageSettings {
    /// Name of the package zone, such as `package-0`.
    pub zone: String,
    pub constraints: Vec<ConstraintSetting>,
}

/// On-disk format of the original snapshot file. `cpus` is indexed by CPU ID.
#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct OriginalFile {
    boot_id: String,
    cpus: Vec<PackageSettings>,
}

/// Where the original settings are kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OriginalStorage {
    /// A file per host boot, named by `original_snapshot_path` from this path.
    Persistent(PathBuf),
    /// Process memory, so every zeusd start records the settings it finds.
    InMemory,
}

/// Return the original settings of this zeusd process according to `storage`.
///
/// `current` holds the settings zeusd finds now. `boot_id_path` is only read
/// for persistent storage.
pub fn establish(
    storage: &OriginalStorage,
    boot_id_path: &Path,
    current: Vec<PackageSettings>,
) -> anyhow::Result<Vec<PackageSettings>> {
    match storage {
        OriginalStorage::InMemory => {
            tracing::info!(
                "Recorded the current CPU power limit settings in memory as the original settings \
                 that reset restores. A restarted Zeusd records the settings it finds at that \
                 time instead."
            );
            Ok(current)
        }
        OriginalStorage::Persistent(path) => {
            let boot_id = read_boot_id(boot_id_path)?;
            load_or_record(path, &boot_id, current)
        }
    }
}

/// Read the kernel's boot ID, a UUID.
pub fn read_boot_id(path: &Path) -> anyhow::Result<String> {
    let contents = fs::read_to_string(path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to read the host boot ID from {}: {e}. Persistent storage of the original CPU \
             power limits needs it to tell host boots apart. {STORAGE_HINT} \
             See {ORIGINAL_POWER_LIMITS_DOC_URL}",
            path.display()
        )
    })?;
    let boot_id = contents.trim();
    if !is_uuid(boot_id) {
        anyhow::bail!(
            "{} contains '{boot_id}', which is not a boot ID. {STORAGE_HINT} \
             See {ORIGINAL_POWER_LIMITS_DOC_URL}",
            path.display()
        );
    }
    Ok(boot_id.to_string())
}

/// Lowercase hyphenated UUID, the format the kernel uses for the boot ID.
fn is_uuid(value: &str) -> bool {
    let groups: Vec<&str> = value.split('-').collect();
    groups.len() == 5
        && groups.iter().zip([8, 4, 4, 4, 12]).all(|(group, len)| {
            group.len() == len
                && group
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        })
}

/// Path of the original snapshot file of the boot `boot_id`.
///
/// The boot ID goes before the extension of the configured file name, or
/// after the file name if it has no extension. For example,
/// `/var/zeusd/original_cpu_power_limit.json` becomes
/// `/var/zeusd/original_cpu_power_limit.<boot_id>.json`.
pub fn original_snapshot_path(path: &Path, boot_id: &str) -> anyhow::Result<PathBuf> {
    let (Some(stem), Some(_)) = (path.file_stem(), path.file_name()) else {
        anyhow::bail!(
            "The original CPU power limit path '{}' does not name a file. Pass a file path to \
             --original-cpu-power-limit-path. See {ORIGINAL_POWER_LIMITS_DOC_URL}",
            path.display()
        );
    };
    let mut name = OsString::from(stem);
    name.push(".");
    name.push(boot_id);
    if let Some(extension) = path.extension() {
        name.push(".");
        name.push(extension);
    }
    Ok(path.with_file_name(name))
}

/// Load the original settings of the boot `boot_id`, or record `current` as
/// them if no zeusd process recorded them yet.
///
/// Errors if the loaded settings do not have the same package zones and
/// constraint names as `current`.
pub fn load_or_record(
    configured_path: &Path,
    boot_id: &str,
    current: Vec<PackageSettings>,
) -> anyhow::Result<Vec<PackageSettings>> {
    let path = original_snapshot_path(configured_path, boot_id)?;
    if record(&path, boot_id, &current)? {
        tracing::info!(
            "Recorded the current CPU power limit settings as the original settings of host boot \
             {boot_id} at {}. Later Zeusd starts in this boot load them if this directory \
             persists across those starts, e.g., a host directory mounted into every Zeusd \
             container.",
            path.display()
        );
        return Ok(current);
    }

    let contents = fs::read_to_string(&path)
        .map_err(|e| storage_error("read the original CPU power limit snapshot", &path, e))?;
    let file: OriginalFile = serde_json::from_str(&contents).map_err(|e| {
        anyhow::anyhow!(
            "Failed to parse the original CPU power limit snapshot at {}: {e}. {REPLACE_HINT} \
             See {ORIGINAL_POWER_LIMITS_DOC_URL}",
            path.display()
        )
    })?;
    if file.boot_id != boot_id {
        anyhow::bail!(
            "The original CPU power limit snapshot at {} records host boot {}, but this is boot {boot_id}. \
             {REPLACE_HINT} See {ORIGINAL_POWER_LIMITS_DOC_URL}",
            path.display(),
            file.boot_id,
        );
    }
    check_matches(&file.cpus, &current).map_err(|e| {
        anyhow::anyhow!(
            "The original CPU power limit snapshot at {} does not match this machine: {e}. \
             {REPLACE_HINT} See {ORIGINAL_POWER_LIMITS_DOC_URL}",
            path.display()
        )
    })?;
    tracing::info!(
        "Loaded the original CPU power limits recorded at the first Zeusd start in host boot \
         {boot_id} from {}",
        path.display()
    );
    Ok(file.cpus)
}

const STORAGE_HINT: &str = "To store the original snapshot elsewhere, pass \
    --original-cpu-power-limit-path; to keep the original settings in memory, which records the \
    settings found at each Zeusd start, pass --no-persistent-original-cpu-power-limit.";

const REPLACE_HINT: &str = "Zeusd does not overwrite the file. To record new original \
    settings, set the CPU power limits to the values reset should restore and delete the file, \
    or reboot.";

/// An error from accessing the storage of the original snapshot, with its likely cause and fixes.
fn storage_error(action: &str, path: &Path, error: std::io::Error) -> anyhow::Error {
    let cause = match error.kind() {
        ErrorKind::ReadOnlyFilesystem => {
            "The filesystem is read-only for Zeusd. Under systemd with ProtectSystem=strict, \
             list the directory in ReadWritePaths=; in a container, mount a writable host \
             directory there. "
        }
        ErrorKind::PermissionDenied => {
            "Zeusd lacks permission for the directory. Make it writable by the user Zeusd runs \
             as. "
        }
        _ => "",
    };
    anyhow::anyhow!(
        "Failed to {action} at {}: {error}. {cause}{STORAGE_HINT} See {ORIGINAL_POWER_LIMITS_DOC_URL}",
        path.display()
    )
}

/// Publish the original snapshot at `path` unless one already exists there, and
/// return whether this call published it.
fn record(path: &Path, boot_id: &str, cpus: &[PackageSettings]) -> anyhow::Result<bool> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                storage_error(
                    "create the directory of the original CPU power limit snapshot",
                    parent,
                    e,
                )
            })?;
        }
    }
    let contents = serde_json::to_string_pretty(&OriginalFile {
        boot_id: boot_id.to_string(),
        cpus: cpus.to_vec(),
    })
    .map_err(|e| {
        anyhow::anyhow!(
            "Failed to serialize the original CPU power limit snapshot: {e}. See {ORIGINAL_POWER_LIMITS_DOC_URL}"
        )
    })?;
    // Processes in different containers can have the same PID, so the time
    // makes their temporary file names differ in the common case.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to name a temporary original CPU power limit snapshot file after the \
                 current time: {e}. Set the system clock to a time after 1970. See {ORIGINAL_POWER_LIMITS_DOC_URL}"
            )
        })?
        .as_nanos();
    publish(
        path,
        contents.as_bytes(),
        &format!("{}.{nanos}", std::process::id()),
    )
}

/// Attempts at finding an unused temporary file name before giving up.
///
/// The limit bounds the work Zeusd does at startup when temporary file names
/// collide, for example with leftover files of earlier Zeusd processes.
const MAX_TMP_ATTEMPTS: u32 = 100;

/// Publish `contents` at `path` unless a file already exists there, and
/// return whether this call published it.
///
/// The contents go to a temporary file named after `writer_id` that is then
/// hard-linked to `path`. Linking fails if `path` exists, so concurrent
/// writers never replace a published file or expose a partially written
/// one. The temporary file is created exclusively, and a name another writer
/// already created is skipped, so writers that share storage and `writer_id`
/// never write to or remove each other's temporary files.
fn publish(path: &Path, contents: &[u8], writer_id: &str) -> anyhow::Result<bool> {
    let (tmp_path, mut file) = create_temporary(path, writer_id)?;
    let written = file
        .write_all(contents)
        .and_then(|()| file.sync_all())
        .map_err(|e| storage_error("write the original CPU power limit snapshot", &tmp_path, e));
    drop(file);
    let published = written.and_then(|()| match fs::hard_link(&tmp_path, path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == ErrorKind::AlreadyExists => Ok(false),
        Err(e) => Err(storage_error(
            "publish the original CPU power limit snapshot with a hard link",
            path,
            e,
        )),
    });
    fs::remove_file(&tmp_path)
        .map_err(|e| storage_error("remove the temporary original snapshot file", &tmp_path, e))?;
    published
}

/// Exclusively create the first unused temporary file name for `path` and
/// `writer_id`.
fn create_temporary(path: &Path, writer_id: &str) -> anyhow::Result<(PathBuf, fs::File)> {
    for attempt in 0..MAX_TMP_ATTEMPTS {
        let mut name = path.as_os_str().to_owned();
        name.push(format!(".tmp.{writer_id}.{attempt}"));
        let tmp_path = PathBuf::from(name);
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)
        {
            Ok(file) => return Ok((tmp_path, file)),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(storage_error(
                    "create a temporary original CPU power limit snapshot file",
                    &tmp_path,
                    e,
                ))
            }
        }
    }
    anyhow::bail!(
        "Failed to create a temporary original CPU power limit snapshot file next to {}: the \
         first {MAX_TMP_ATTEMPTS} names for this Zeusd process already exist. Remove leftover \
         '{}.tmp.*' files that no running Zeusd is writing. See {ORIGINAL_POWER_LIMITS_DOC_URL}",
        path.display(),
        path.display(),
    )
}

fn check_matches(recorded: &[PackageSettings], current: &[PackageSettings]) -> anyhow::Result<()> {
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
        let names = |package: &PackageSettings| -> Vec<String> {
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

    const BOOT_A: &str = "8f3e2c1a-5b6d-4e7f-9a0b-1c2d3e4f5a6b";
    const BOOT_B: &str = "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d";

    fn package(zone: &str, settings: &[(&str, u64, Option<u64>)]) -> PackageSettings {
        PackageSettings {
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

    fn intel(long_term_uw: u64) -> Vec<PackageSettings> {
        vec![package(
            "package-0",
            &[
                ("long_term", long_term_uw, Some(999_424)),
                ("short_term", 246_000_000, Some(2_440)),
            ],
        )]
    }

    fn file_names(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn boot_id_goes_before_the_extension() {
        assert_eq!(
            original_snapshot_path(
                Path::new("/var/zeusd/original_cpu_power_limit.json"),
                BOOT_A
            )
            .unwrap(),
            PathBuf::from(format!("/var/zeusd/original_cpu_power_limit.{BOOT_A}.json"))
        );
        assert_eq!(
            original_snapshot_path(Path::new("/srv/original"), BOOT_A).unwrap(),
            PathBuf::from(format!("/srv/original.{BOOT_A}"))
        );
        assert_eq!(
            original_snapshot_path(Path::new("original.v1.json"), BOOT_A).unwrap(),
            PathBuf::from(format!("original.v1.{BOOT_A}.json"))
        );
        assert!(original_snapshot_path(Path::new("/"), BOOT_A).is_err());
    }

    #[test]
    fn reads_and_validates_boot_id() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("boot_id");
        fs::write(&path, format!("{BOOT_A}\n")).unwrap();
        assert_eq!(read_boot_id(&path).unwrap(), BOOT_A);

        for invalid in ["", "not-a-uuid", "8F3E2C1A-5B6D-4E7F-9A0B-1C2D3E4F5A6B"] {
            fs::write(&path, invalid).unwrap();
            assert!(read_boot_id(&path).is_err(), "{invalid:?}");
        }

        let message = read_boot_id(&tmp.path().join("absent"))
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("--no-persistent-original-cpu-power-limit"),
            "{message}"
        );
        assert!(message.contains(ORIGINAL_POWER_LIMITS_DOC_URL), "{message}");
    }

    #[test]
    fn same_boot_reuses_the_recorded_original() {
        let tmp = tempfile::tempdir().unwrap();
        let configured = tmp.path().join("zeusd").join("original.json");

        assert_eq!(
            load_or_record(&configured, BOOT_A, intel(205_000_000)).unwrap(),
            intel(205_000_000)
        );
        assert!(!configured.exists());
        assert_eq!(
            file_names(configured.parent().unwrap()),
            vec![format!("original.{BOOT_A}.json")]
        );

        // A restarted or replaced zeusd sees changed limits but keeps the recorded original.
        assert_eq!(
            load_or_record(&configured, BOOT_A, intel(150_000_000)).unwrap(),
            intel(205_000_000)
        );
    }

    #[test]
    fn new_boot_records_the_current_settings() {
        let tmp = tempfile::tempdir().unwrap();
        let configured = tmp.path().join("original.json");
        load_or_record(&configured, BOOT_A, intel(205_000_000)).unwrap();

        assert_eq!(
            load_or_record(&configured, BOOT_B, intel(150_000_000)).unwrap(),
            intel(150_000_000)
        );
        assert_eq!(
            load_or_record(&configured, BOOT_A, intel(100_000_000)).unwrap(),
            intel(205_000_000)
        );
        assert_eq!(
            file_names(tmp.path()),
            vec![
                format!("original.{BOOT_B}.json"),
                format!("original.{BOOT_A}.json"),
            ]
        );
    }

    #[test]
    fn mismatched_original_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("original.json");
        let boot = vec![package("package-0", &[("socket", 200_000_000, None)])];
        load_or_record(&path, BOOT_A, boot).unwrap();

        let more_packages = vec![
            package("package-0", &[("socket", 200_000_000, None)]),
            package("package-1", &[("socket", 200_000_000, None)]),
        ];
        assert!(load_or_record(&path, BOOT_A, more_packages).is_err());

        let other_zone = vec![package("package-1", &[("socket", 200_000_000, None)])];
        assert!(load_or_record(&path, BOOT_A, other_zone).is_err());

        let other_constraints = vec![package("package-0", &[])];
        let message = load_or_record(&path, BOOT_A, other_constraints)
            .unwrap_err()
            .to_string();
        assert!(message.contains("does not match this machine"), "{message}");
        assert!(message.contains(ORIGINAL_POWER_LIMITS_DOC_URL), "{message}");
    }

    #[test]
    fn original_of_another_boot_under_this_boots_name_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let configured = tmp.path().join("original.json");
        load_or_record(&configured, BOOT_A, intel(205_000_000)).unwrap();
        fs::rename(
            original_snapshot_path(&configured, BOOT_A).unwrap(),
            original_snapshot_path(&configured, BOOT_B).unwrap(),
        )
        .unwrap();

        let message = load_or_record(&configured, BOOT_B, intel(150_000_000))
            .unwrap_err()
            .to_string();
        assert!(
            message.contains(&format!("records host boot {BOOT_A}")),
            "{message}"
        );
        assert!(
            fs::read_to_string(original_snapshot_path(&configured, BOOT_B).unwrap())
                .unwrap()
                .contains(BOOT_A)
        );
    }

    #[test]
    fn corrupt_original_errors_and_is_kept() {
        let tmp = tempfile::tempdir().unwrap();
        let configured = tmp.path().join("original.json");
        let path = original_snapshot_path(&configured, BOOT_A).unwrap();
        for corrupt in ["", "{", "{\"cpus\": []}", "{\"boot_id\": 1, \"cpus\": []}"] {
            fs::write(&path, corrupt).unwrap();
            let message = load_or_record(&configured, BOOT_A, vec![])
                .unwrap_err()
                .to_string();
            assert!(message.contains("Failed to parse"), "{message}");
            assert!(message.contains("delete the file"), "{message}");
            assert_eq!(fs::read_to_string(&path).unwrap(), corrupt);
        }
    }

    #[cfg(unix)]
    #[test]
    fn unwritable_directory_explains_the_storage_options() {
        use std::os::unix::fs::PermissionsExt;

        if nix::unistd::geteuid().is_root() {
            // Root ignores file permissions, so no write can be made to fail.
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("zeusd");
        fs::create_dir(&dir).unwrap();
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();

        let message = load_or_record(&dir.join("original.json"), BOOT_A, intel(205_000_000))
            .unwrap_err()
            .to_string();
        assert!(message.contains("lacks permission"), "{message}");
        assert!(
            message.contains("--original-cpu-power-limit-path"),
            "{message}"
        );
        assert!(
            message.contains("--no-persistent-original-cpu-power-limit"),
            "{message}"
        );
        assert!(message.contains(ORIGINAL_POWER_LIMITS_DOC_URL), "{message}");
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[test]
    fn concurrent_starts_agree_on_one_original() {
        let tmp = tempfile::tempdir().unwrap();
        let configured = tmp.path().join("original.json");

        let results: Vec<Vec<PackageSettings>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..16u64)
                .map(|i| {
                    let configured = &configured;
                    scope.spawn(move || {
                        let current =
                            vec![package("package-0", &[("socket", 100_000_000 + i, None)])];
                        load_or_record(configured, BOOT_A, current).unwrap()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        assert!(results.iter().all(|r| *r == results[0]));
        assert_eq!(
            load_or_record(&configured, BOOT_A, results[0].clone()).unwrap(),
            results[0]
        );
        assert_eq!(
            file_names(tmp.path()),
            vec![format!("original.{BOOT_A}.json")]
        );
    }

    /// Two containers whose Zeusd processes are both PID 1 can pick the same
    /// temporary file name in shared storage.
    #[test]
    fn temporary_file_of_another_writer_with_the_same_id_is_kept() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("original.json");
        let other = tmp.path().join("original.json.tmp.1.0");
        fs::write(&other, "partial original snapshot of another container").unwrap();

        assert!(publish(&path, b"ours", "1").unwrap());
        assert_eq!(fs::read_to_string(&path).unwrap(), "ours");
        assert_eq!(
            fs::read_to_string(&other).unwrap(),
            "partial original snapshot of another container"
        );
        assert_eq!(
            file_names(tmp.path()),
            vec!["original.json", "original.json.tmp.1.0"]
        );

        assert!(!publish(&path, b"later", "1").unwrap());
        assert_eq!(fs::read_to_string(&path).unwrap(), "ours");
    }

    #[test]
    fn concurrent_writers_with_the_same_id_publish_one_complete_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("original.json");
        let writers = 16;
        let barrier = std::sync::Barrier::new(writers);

        let published: Vec<(String, bool)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..writers)
                .map(|i| {
                    let (path, barrier) = (&path, &barrier);
                    scope.spawn(move || {
                        let contents = format!("writer {i}");
                        barrier.wait();
                        let published = publish(path, contents.as_bytes(), "1").unwrap();
                        (contents, published)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let winners: Vec<&String> = published
            .iter()
            .filter(|(_, published)| *published)
            .map(|(contents, _)| contents)
            .collect();
        assert_eq!(winners.len(), 1, "{published:?}");
        assert_eq!(fs::read_to_string(&path).unwrap(), *winners[0]);
        assert_eq!(file_names(tmp.path()), vec!["original.json"]);
    }

    #[test]
    fn exhausted_temporary_names_error_without_touching_them() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("original.json");
        for attempt in 0..MAX_TMP_ATTEMPTS {
            fs::write(
                tmp.path().join(format!("original.json.tmp.1.{attempt}")),
                "",
            )
            .unwrap();
        }

        let message = publish(&path, b"ours", "1").unwrap_err().to_string();
        assert!(message.contains("Remove leftover"), "{message}");
        assert!(message.contains(ORIGINAL_POWER_LIMITS_DOC_URL), "{message}");
        assert!(!path.exists());
        assert_eq!(file_names(tmp.path()).len(), MAX_TMP_ATTEMPTS as usize);
    }

    #[test]
    fn persistent_storage_reads_the_boot_id() {
        let tmp = tempfile::tempdir().unwrap();
        let boot_id_path = tmp.path().join("boot_id");
        let storage = OriginalStorage::Persistent(tmp.path().join("state").join("original.json"));

        fs::write(&boot_id_path, format!("{BOOT_A}\n")).unwrap();
        establish(&storage, &boot_id_path, intel(205_000_000)).unwrap();
        assert_eq!(
            establish(&storage, &boot_id_path, intel(150_000_000)).unwrap(),
            intel(205_000_000)
        );

        fs::write(&boot_id_path, format!("{BOOT_B}\n")).unwrap();
        assert_eq!(
            establish(&storage, &boot_id_path, intel(150_000_000)).unwrap(),
            intel(150_000_000)
        );
        assert!(establish(&storage, &tmp.path().join("absent"), intel(1)).is_err());
    }

    #[test]
    fn in_memory_storage_records_every_start_without_files() {
        let tmp = tempfile::tempdir().unwrap();
        let absent_boot_id = tmp.path().join("absent");

        for long_term_uw in [205_000_000, 150_000_000] {
            assert_eq!(
                establish(
                    &OriginalStorage::InMemory,
                    &absent_boot_id,
                    intel(long_term_uw)
                )
                .unwrap(),
                intel(long_term_uw)
            );
        }
        assert!(file_names(tmp.path()).is_empty());
    }
}
