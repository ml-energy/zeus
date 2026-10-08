//! AMD Host System Management Port (HSMP) access through `/dev/hsmp`.
//!
//! HSMP is a mailbox to the System Management Unit (SMU) firmware of AMD EPYC
//! CPUs. The kernel's `amd_hsmp` driver exposes it as an ioctl on `/dev/hsmp`.
//! Only the socket power limit messages are used here.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;
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

/// Sends HSMP messages to the SMU firmware.
pub trait HsmpTransport: Send + Sync {
    /// Send `msg`. If the message has a response, the kernel writes it to `msg.args`.
    fn send(&self, msg: &mut HsmpMessage) -> io::Result<()>;
}

/// The HSMP character device.
pub struct HsmpDevice {
    file: File,
}

impl HsmpDevice {
    /// Open the HSMP device, or return `None` if it does not exist.
    ///
    /// The kernel accepts set messages only through a file opened for writing
    /// and get messages only through a file opened for reading.
    pub fn open(path: &Path, writable: bool) -> io::Result<Option<Self>> {
        match OpenOptions::new().read(true).write(writable).open(path) {
            Ok(file) => Ok(Some(Self { file })),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }
}

impl HsmpTransport for HsmpDevice {
    #[cfg(target_os = "linux")]
    fn send(&self, msg: &mut HsmpMessage) -> io::Result<()> {
        use std::os::fd::AsRawFd;
        // SAFETY: `msg` is a valid `struct hsmp_message` that is exclusively
        // borrowed for the duration of the call.
        unsafe { hsmp_ioctl(self.file.as_raw_fd(), msg) }
            .map(|_| ())
            .map_err(io::Error::from)
    }

    #[cfg(not(target_os = "linux"))]
    fn send(&self, _msg: &mut HsmpMessage) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "HSMP is only available on Linux",
        ))
    }
}

/// One CPU socket addressed through HSMP.
#[derive(Clone)]
pub struct HsmpSocket {
    transport: Arc<dyn HsmpTransport>,
    sock_ind: u16,
}

impl HsmpSocket {
    pub fn new(transport: Arc<dyn HsmpTransport>, sock_ind: u16) -> Self {
        Self {
            transport,
            sock_ind,
        }
    }

    /// Read the socket power limit in milliwatts.
    pub fn power_limit_mw(&self) -> io::Result<u32> {
        self.get(HSMP_GET_SOCKET_POWER_LIMIT)
    }

    /// Read the largest socket power limit the firmware accepts, in milliwatts.
    pub fn power_limit_max_mw(&self) -> io::Result<u32> {
        self.get(HSMP_GET_SOCKET_POWER_LIMIT_MAX)
    }

    /// Set the socket power limit in milliwatts.
    ///
    /// The firmware clamps values above `power_limit_max_mw` instead of
    /// rejecting them.
    pub fn set_power_limit_mw(&self, power_limit_mw: u32) -> io::Result<()> {
        let mut msg = HsmpMessage {
            msg_id: HSMP_SET_SOCKET_POWER_LIMIT,
            num_args: 1,
            response_sz: 0,
            sock_ind: self.sock_ind,
            ..Default::default()
        };
        msg.args[0] = power_limit_mw;
        self.transport.send(&mut msg)
    }

    fn get(&self, msg_id: u32) -> io::Result<u32> {
        let mut msg = HsmpMessage {
            msg_id,
            num_args: 0,
            response_sz: 1,
            sock_ind: self.sock_ind,
            ..Default::default()
        };
        self.transport.send(&mut msg)?;
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
        fn send(&self, msg: &mut HsmpMessage) -> io::Result<()> {
            self.sent.lock().unwrap().push(msg.clone());
            let mut limits = self.limits_mw.lock().unwrap();
            let socket = limits
                .get_mut(msg.sock_ind as usize)
                .ok_or_else(|| io::Error::from_raw_os_error(19))?;
            match msg.msg_id {
                HSMP_SET_SOCKET_POWER_LIMIT => *socket = msg.args[0].min(self.max_mw),
                HSMP_GET_SOCKET_POWER_LIMIT => msg.args[0] = *socket,
                HSMP_GET_SOCKET_POWER_LIMIT_MAX => msg.args[0] = self.max_mw,
                _ => return Err(io::Error::from_raw_os_error(42)),
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
    fn open_missing_device_is_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(HsmpDevice::open(&tmp.path().join("hsmp"), true)
            .unwrap()
            .is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires an AMD EPYC CPU with the amd_hsmp kernel module loaded"]
    fn reads_socket_power_limit_from_device() {
        let device = HsmpDevice::open(Path::new(HSMP_DEVICE_PATH), false)
            .unwrap()
            .expect("HSMP device does not exist");
        let socket = HsmpSocket::new(Arc::new(device), 0);
        let limit = socket.power_limit_mw().unwrap();
        let max = socket.power_limit_max_mw().unwrap();
        assert!(limit > 0 && limit <= max, "limit {limit} mW, max {max} mW");
    }
}
