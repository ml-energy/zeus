use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const LIBRARY_NAME: &str = "libamd_smi.so";

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct LibraryVersion {
    major: u32,
    minor: u32,
    patch: Option<u32>,
}

impl LibraryVersion {
    fn parse(path: &Path) -> Option<Self> {
        let file_name = path.file_name()?.to_str()?;
        let version = file_name.strip_prefix("libamd_smi.so.")?;
        let mut components = version.split('.');
        let major = components.next()?.parse().ok()?;
        let minor = components.next()?.parse().ok()?;
        let patch = components.next().map(str::parse).transpose().ok()?;
        if components.next().is_some() {
            return None;
        }
        Some(Self {
            major,
            minor,
            patch,
        })
    }

    fn display(&self) -> String {
        match self.patch {
            Some(patch) => format!("{}.{}.{}", self.major, self.minor, patch),
            None => format!("{}.{}", self.major, self.minor),
        }
    }
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(amdsmi_abi, values(\"24\", \"25\", \"26\"))");
    println!("cargo:rerun-if-env-changed=AMDSMI_LIB_DIR");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");

    if env::var_os("CARGO_FEATURE_AMDSMI").is_none() {
        return;
    }

    let library_dir = locate_library_dir().unwrap_or_else(|| {
        panic!(
            "could not locate {LIBRARY_NAME}; set ROCM_PATH to the ROCm installation root or \
             AMDSMI_LIB_DIR to the directory containing {LIBRARY_NAME}"
        )
    });
    let version = find_library_version(&library_dir).unwrap_or_else(|| {
        panic!(
            "could not find a fully versioned {LIBRARY_NAME} in {}; set ROCM_PATH to the ROCm \
             installation root or AMDSMI_LIB_DIR to the directory containing {LIBRARY_NAME}",
            library_dir.display()
        )
    });

    validate_version(&version);

    println!("cargo:rustc-link-search=native={}", library_dir.display());
    println!("cargo:rustc-link-lib=dylib=amd_smi");
    println!("cargo:rustc-cfg=amdsmi_abi=\"{}\"", version.major);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", library_dir.display());
    println!(
        "cargo:rustc-env=ZEUSD_AMDSMI_BUILD_VERSION={}",
        version.display()
    );
}

fn locate_library_dir() -> Option<PathBuf> {
    if let Some(path) = env::var_os("AMDSMI_LIB_DIR") {
        return Some(PathBuf::from(path));
    }
    if let Some(path) = env::var_os("ROCM_PATH") {
        return Some(PathBuf::from(path).join("lib"));
    }

    let default_rocm = PathBuf::from("/opt/rocm");
    if default_rocm.exists() {
        return Some(default_rocm.join("lib"));
    }

    fs::read_dir("/opt")
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("rocm-"))
        })
        .max()
        .map(|path| path.join("lib"))
}

fn find_library_version(library_dir: &Path) -> Option<LibraryVersion> {
    let unversioned = library_dir.join(LIBRARY_NAME);
    if fs::symlink_metadata(&unversioned).is_ok() {
        if let Ok(resolved) = fs::canonicalize(&unversioned) {
            if let Some(version) = LibraryVersion::parse(&resolved) {
                return Some(version);
            }
        }
    }

    fs::read_dir(library_dir)
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(LIBRARY_NAME))
        })
        .filter_map(|entry| fs::canonicalize(entry.path()).ok())
        .filter_map(|path| LibraryVersion::parse(&path))
        .max()
}

fn validate_version(version: &LibraryVersion) {
    match (version.major, version.minor) {
        (24, 6) => {
            panic!("found libamd_smi.so 24.6 (ROCm 6.2); zeusd supports ROCm 6.3 and newer")
        }
        (24, 7..) | (25, _) | (26, _) => {}
        _ => panic!("unsupported amd-smi ABI; this zeusd release supports ROCm 6.3 through 7.2"),
    }
}
