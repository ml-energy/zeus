//! AMD Host System Management Port (HSMP) access through `/dev/hsmp`.
//!
//! HSMP is a mailbox to the System Management Unit (SMU) firmware of AMD EPYC
//! CPUs. The kernel's `amd_hsmp` driver exposes it as an ioctl on `/dev/hsmp`.
//! Only the socket power limit messages are used here.

#[cfg(target_os = "linux")]
use std::fs::OpenOptions;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Path of the HSMP character device created by the `amd_hsmp` kernel driver.
pub const HSMP_DEVICE_PATH: &str = "/dev/hsmp";

const HSMP_MAX_MSG_LEN: usize = 8;
const HSMP_SET_SOCKET_POWER_LIMIT: u32 = 0x05;
const HSMP_GET_SOCKET_POWER_LIMIT: u32 = 0x06;
const HSMP_GET_SOCKET_POWER_LIMIT_MAX: u32 = 0x07;

/// `struct hsmp_message` from the kernel header `arch/x86/include/uapi/asm/amd_hsmp.h`.
#[repr(C)]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct HsmpMessage {
    pub msg_id: u32,
    pub num_args: u16,
    pub response_sz: u16,
    pub args: [u32; HSMP_MAX_MSG_LEN],
    pub sock_ind: u16,
}

// The ioctl number encodes the struct size, so a layout mismatch with the
// kernel would make every request fail with `ENOTTY`.
const _: () = assert!(std::mem::size_of::<HsmpMessage>() == 44);

#[cfg(target_os = "linux")]
nix::ioctl_readwrite!(hsmp_ioctl, 0xF8, 0, HsmpMessage);

/// Why an HSMP operation failed, with the action that fixes it.
#[derive(thiserror::Error, Debug)]
pub enum HsmpError {
    #[error(
        "{0} does not exist. Load the amd_hsmp kernel module with `sudo modprobe amd_hsmp` \
         (in a container, also pass the device) and restart Zeusd."
    )]
    DeviceMissing(PathBuf),
    #[error(
        "Opening {path} for {} was denied: {source}. HSMP queries need read permission on the \
         device and HSMP setting changes need write permission; the device is owned by root. \
         In a container, the device must also be passed and allowed by the device cgroup.",
        if *.write { "writing" } else { "reading" }
    )]
    PermissionDenied {
        path: PathBuf,
        write: bool,
        source: io::Error,
    },
    #[error("Failed to open {path}: {source}")]
    Open { path: PathBuf, source: io::Error },
    #[error("The HSMP request failed: {0}{}", request_hint(.0))]
    Request(io::Error),
}

/// Explain a failed HSMP ioctl on a device that opened.
fn request_hint(source: &io::Error) -> &'static str {
    match source.kind() {
        io::ErrorKind::PermissionDenied => {
            ". The device opened, but the kernel denied the ioctl on it. Check device access \
             and the security policy applied to Zeusd or its container, including SELinux, \
             AppArmor, and seccomp. These policies must permit the HSMP ioctl."
        }
        _ => "",
    }
}

/// Sends HSMP messages to the SMU firmware.
pub trait HsmpTransport: Send + Sync {
    /// Send `msg`, which changes a setting if `write` is true and only reads
    /// one otherwise. If the message has a response, the kernel writes it to
    /// `msg.args`.
    fn send(&self, msg: &mut HsmpMessage, write: bool) -> Result<(), HsmpError>;
}

/// The HSMP character device.
///
/// The device is opened for each message: read-only for get messages and
/// write-only for set messages, which is the access the kernel requires for
/// each. HSMP queries therefore work without write permission.
pub struct HsmpDevice {
    path: PathBuf,
}

impl HsmpDevice {
    /// Return the device at `path`, or `None` if nothing exists there.
    ///
    /// Whether the device can be opened is checked on each message.
    pub fn find(path: &Path) -> Option<Self> {
        match std::fs::symlink_metadata(path) {
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            _ => Some(Self {
                path: path.to_path_buf(),
            }),
        }
    }

    /// Path of the device.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Open the device for reading or writing to check whether that access is granted.
    pub fn check_access(&self, write: bool) -> Result<(), HsmpError> {
        self.open(write).map(drop)
    }

    fn open(&self, write: bool) -> Result<std::fs::File, HsmpError> {
        #[cfg(target_os = "linux")]
        {
            OpenOptions::new()
                .read(!write)
                .write(write)
                .open(&self.path)
                .map_err(|source| match source.kind() {
                    io::ErrorKind::NotFound => HsmpError::DeviceMissing(self.path.clone()),
                    io::ErrorKind::PermissionDenied => HsmpError::PermissionDenied {
                        path: self.path.clone(),
                        write,
                        source,
                    },
                    _ => HsmpError::Open {
                        path: self.path.clone(),
                        source,
                    },
                })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = write;
            Err(HsmpError::DeviceMissing(self.path.clone()))
        }
    }
}

impl HsmpTransport for HsmpDevice {
    fn send(&self, msg: &mut HsmpMessage, write: bool) -> Result<(), HsmpError> {
        let file = self.open(write)?;
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: `msg` is a valid `struct hsmp_message` that is exclusively
            // borrowed for the duration of the call.
            unsafe { hsmp_ioctl(file.as_raw_fd(), msg) }
                .map(|_| ())
                .map_err(|e| HsmpError::Request(io::Error::from(e)))
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (file, msg);
            Err(HsmpError::DeviceMissing(self.path.clone()))
        }
    }
}

/// One CPU socket addressed through HSMP.
#[derive(Clone)]
pub struct HsmpSocket {
    transport: Arc<dyn HsmpTransport>,
    sock_ind: u16,
}

impl HsmpSocket {
    /// Address the socket whose HSMP socket index is `sock_ind`, which is the
    /// physical package ID and not necessarily the Zeusd CPU index.
    pub fn new(transport: Arc<dyn HsmpTransport>, sock_ind: u16) -> Self {
        Self {
            transport,
            sock_ind,
        }
    }

    /// Read the socket power limit in milliwatts.
    pub fn power_limit_mw(&self) -> Result<u32, HsmpError> {
        self.get(HSMP_GET_SOCKET_POWER_LIMIT)
    }

    /// Read the largest socket power limit the firmware accepts, in milliwatts.
    pub fn power_limit_max_mw(&self) -> Result<u32, HsmpError> {
        self.get(HSMP_GET_SOCKET_POWER_LIMIT_MAX)
    }

    /// Set the socket power limit in milliwatts.
    ///
    /// The firmware clamps values above `power_limit_max_mw` instead of
    /// rejecting them.
    pub fn set_power_limit_mw(&self, power_limit_mw: u32) -> Result<(), HsmpError> {
        let mut msg = HsmpMessage {
            msg_id: HSMP_SET_SOCKET_POWER_LIMIT,
            num_args: 1,
            response_sz: 0,
            sock_ind: self.sock_ind,
            ..Default::default()
        };
        msg.args[0] = power_limit_mw;
        self.transport.send(&mut msg, true)
    }

    fn get(&self, msg_id: u32) -> Result<u32, HsmpError> {
        let mut msg = HsmpMessage {
            msg_id,
            num_args: 0,
            response_sz: 1,
            sock_ind: self.sock_ind,
            ..Default::default()
        };
        self.transport.send(&mut msg, false)?;
        Ok(msg.args[0])
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::Mutex;

    /// An in-memory HSMP firmware with one socket power limit per socket.
    pub(crate) struct FakeHsmp {
        pub limits_mw: Mutex<Vec<u32>>,
        pub max_mw: u32,
        pub sent: Mutex<Vec<HsmpMessage>>,
    }

    impl FakeHsmp {
        pub(crate) fn new(limits_mw: Vec<u32>, max_mw: u32) -> Arc<Self> {
            Arc::new(Self {
                limits_mw: Mutex::new(limits_mw),
                max_mw,
                sent: Mutex::new(Vec::new()),
            })
        }
    }

    impl HsmpTransport for FakeHsmp {
        fn send(&self, msg: &mut HsmpMessage, write: bool) -> Result<(), HsmpError> {
            assert_eq!(write, msg.msg_id == HSMP_SET_SOCKET_POWER_LIMIT);
            self.sent.lock().unwrap().push(msg.clone());
            let mut limits = self.limits_mw.lock().unwrap();
            let socket = limits
                .get_mut(msg.sock_ind as usize)
                .ok_or_else(|| HsmpError::Request(io::Error::from_raw_os_error(19)))?;
            match msg.msg_id {
                HSMP_SET_SOCKET_POWER_LIMIT => *socket = msg.args[0].min(self.max_mw),
                HSMP_GET_SOCKET_POWER_LIMIT => msg.args[0] = *socket,
                HSMP_GET_SOCKET_POWER_LIMIT_MAX => msg.args[0] = self.max_mw,
                _ => return Err(HsmpError::Request(io::Error::from_raw_os_error(42))),
            }
            Ok(())
        }
    }

    #[test]
    fn get_messages_request_one_response_word() {
        let fake = FakeHsmp::new(vec![200_000, 225_000], 280_000);
        let socket = HsmpSocket::new(fake.clone(), 1);

        assert_eq!(socket.power_limit_mw().unwrap(), 225_000);
        assert_eq!(socket.power_limit_max_mw().unwrap(), 280_000);

        let sent = fake.sent.lock().unwrap();
        assert_eq!(sent.len(), 2);
        for (msg, msg_id) in sent
            .iter()
            .zip([HSMP_GET_SOCKET_POWER_LIMIT, HSMP_GET_SOCKET_POWER_LIMIT_MAX])
        {
            assert_eq!(msg.msg_id, msg_id);
            assert_eq!(msg.num_args, 0);
            assert_eq!(msg.response_sz, 1);
            assert_eq!(msg.sock_ind, 1);
        }
    }

    #[test]
    fn set_message_carries_one_argument() {
        let fake = FakeHsmp::new(vec![200_000], 280_000);
        let socket = HsmpSocket::new(fake.clone(), 0);

        socket.set_power_limit_mw(150_000).unwrap();

        let sent = fake.sent.lock().unwrap();
        assert_eq!(
            sent[0],
            HsmpMessage {
                msg_id: HSMP_SET_SOCKET_POWER_LIMIT,
                num_args: 1,
                response_sz: 0,
                args: [150_000, 0, 0, 0, 0, 0, 0, 0],
                sock_ind: 0,
            }
        );
        assert_eq!(fake.limits_mw.lock().unwrap()[0], 150_000);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn ioctl_number_matches_kernel() {
        // `_IOWR(0xF8, 0, struct hsmp_message)` with a 44-byte struct.
        assert_eq!(
            nix::request_code_readwrite!(0xF8, 0, std::mem::size_of::<HsmpMessage>()),
            0xC02C_F800
        );
    }

    #[test]
    fn find_missing_device_is_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(HsmpDevice::find(&tmp.path().join("hsmp")).is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn device_removed_after_startup_explains_how_to_load_driver() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("hsmp");
        std::fs::write(&path, "").unwrap();
        let socket = HsmpSocket::new(Arc::new(HsmpDevice::find(&path).unwrap()), 0);
        std::fs::remove_file(&path).unwrap();

        let error = socket.power_limit_mw().unwrap_err();
        assert!(matches!(error, HsmpError::DeviceMissing(_)), "{error}");
        assert!(error.to_string().contains("modprobe amd_hsmp"), "{error}");
    }

    /// Reads open the device read-only, so they need no write permission, and
    /// writes are never retried read-only.
    #[cfg(target_os = "linux")]
    #[test]
    fn reads_and_writes_open_the_device_with_matching_access() {
        use std::os::unix::fs::PermissionsExt;

        if nix::unistd::geteuid().is_root() {
            // Root ignores file permissions, so no open can be made to fail.
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("hsmp");
        std::fs::write(&path, "").unwrap();
        let device = HsmpDevice::find(&path).unwrap();
        let socket = HsmpSocket::new(Arc::new(HsmpDevice::find(&path).unwrap()), 0);

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).unwrap();
        device.check_access(false).unwrap();
        // A regular file opens but rejects the ioctl.
        assert!(matches!(
            socket.power_limit_mw(),
            Err(HsmpError::Request(_))
        ));
        let error = socket.set_power_limit_mw(150_000).unwrap_err();
        assert!(
            matches!(error, HsmpError::PermissionDenied { write: true, .. }),
            "{error}"
        );
        assert!(error.to_string().contains("for writing"), "{error}");

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o222)).unwrap();
        device.check_access(true).unwrap();
        assert!(matches!(
            socket.power_limit_mw(),
            Err(HsmpError::PermissionDenied { write: false, .. })
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires an AMD EPYC CPU with the amd_hsmp kernel module loaded"]
    fn reads_socket_power_limit_from_device() {
        let device =
            HsmpDevice::find(Path::new(HSMP_DEVICE_PATH)).expect("HSMP device does not exist");
        let socket = HsmpSocket::new(Arc::new(device), 0);
        let limit = socket.power_limit_mw().unwrap();
        let max = socket.power_limit_max_mw().unwrap();
        assert!(limit > 0 && limit <= max, "limit {limit} mW, max {max} mW");
    }
}
