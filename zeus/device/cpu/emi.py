"""Windows EMI (Energy Meter Interface) CPU energy monitoring.

- EMI (Energy Meter Interface):
   EMI is a Windows interface introduced in Windows 10 that allows applications to read energy
   consumption data from hardware energy meters. It provides access to RAPL (Running Average Power
   Limit) counters on Intel processors via a standardized IOCTL interface.

- Energy Meter Device:
   An EMI device represents one energy metering unit. On Intel systems, a single EMI device
   typically exposes multiple channels corresponding to different power domains (e.g., package,
   DRAM, PP0, PP1) for each CPU socket.

- Channel:
   Each EMI device exposes one or more named channels. Channel names follow the pattern
   ``RAPL_Package{N}_{DOMAIN}`` where N is the socket index and DOMAIN is the power domain
   (e.g., PKG, DRAM, PP0, PP1).

See: https://learn.microsoft.com/en-us/windows-hardware/drivers/powermeter/energy-meter-interface
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import re
import sys
from functools import lru_cache
from typing import Sequence

import zeus.device.cpu.common as cpu_common
from zeus.device.cpu.common import CpuDramMeasurement
from zeus.device.exception import ZeusBaseCPUError

logger = logging.getLogger(__name__)

# EMI is a Windows-only interface. windll/wintypes are only available on Windows.
_WINDOWS = sys.platform == "win32"

if _WINDOWS:
    from ctypes import windll, wintypes

    # -----------------------------------------------------------------------
    # Constants
    # -----------------------------------------------------------------------

    # GUID for the EMI device interface: {45BD8344-7ED6-49CF-A440-C276C933B053}
    _EMI_GUID_DATA1: int = 0x45BD8344
    _EMI_GUID_DATA2: int = 0x7ED6
    _EMI_GUID_DATA3: int = 0x49CF
    _EMI_GUID_DATA4: tuple[int, ...] = (0xA4, 0x40, 0xC2, 0x76, 0xC9, 0x33, 0xB0, 0x53)

    # IOCTLs — CTL_CODE(FILE_DEVICE_UNKNOWN=0x22, func, METHOD_BUFFERED=0, FILE_READ_ACCESS=1)
    # = (0x22 << 16) | (1 << 14) | (func << 2) | 0
    _IOCTL_EMI_GET_VERSION: int = 0x00224000  # func = 0
    _IOCTL_EMI_GET_METADATA_SIZE: int = 0x00224004  # func = 1
    _IOCTL_EMI_GET_METADATA: int = 0x00224008  # func = 2
    _IOCTL_EMI_GET_MEASUREMENT: int = 0x0022400C  # func = 3

    _EMI_VERSION_V1: int = 1
    _EMI_VERSION_V2: int = 2

    # EMI_NAME_MAX: maximum WCHAR count for OEM/Model fields in metadata (= 16 WCHARs = 32 bytes)
    _EMI_NAME_MAX: int = 16

    # EmiMeasurementUnitPicowattHours = 0
    _EMI_MEASUREMENT_UNIT_PICOWATT_HOURS: int = 0

    # 1 picowatt-hour = 1e-12 W * 3600 s = 3.6e-9 J = 3.6e-6 mJ
    _PICOWATT_HOURS_TO_MILLIJOULES: float = 3.6e-6

    _GENERIC_READ: int = 0x80000000
    _FILE_SHARE_READ: int = 0x00000001
    _OPEN_EXISTING: int = 3
    # INVALID_HANDLE_VALUE: all bits set, platform-width.
    # Using bit arithmetic avoids the Optional[int] return type of ctypes.c_void_p.value.
    _INVALID_HANDLE_VALUE: int = 2 ** (ctypes.sizeof(ctypes.c_void_p) * 8) - 1
    _DIGCF_PRESENT: int = 0x00000002
    _DIGCF_DEVICEINTERFACE: int = 0x00000010

    # cbSize value for SP_DEVICE_INTERFACE_DETAIL_DATA_W.
    # The struct contains DWORD cbSize (4 bytes) + WCHAR DevicePath[1] (2 bytes).
    # MSVC pads the struct to the largest-member alignment (DWORD = 4 bytes),
    # making the total sizeof() = 8 on both 32- and 64-bit Windows.
    _DETAIL_DATA_CBSIZE: int = 8

    # -----------------------------------------------------------------------
    # ctypes structures
    # -----------------------------------------------------------------------

    class _Guid(ctypes.Structure):  # noqa: N801 (Windows API name)
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    class _SpDeviceInterfaceData(ctypes.Structure):  # noqa: N801 (Windows API name)
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("InterfaceClassGuid", _Guid),
            ("Flags", wintypes.DWORD),
            ("Reserved", ctypes.c_size_t),
        ]

    # -----------------------------------------------------------------------
    # Windows API setup
    # -----------------------------------------------------------------------

    _setupapi = windll.setupapi
    _kernel32 = windll.kernel32

    _setupapi.SetupDiGetClassDevsW.restype = ctypes.c_void_p
    _setupapi.SetupDiGetClassDevsW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    _setupapi.SetupDiEnumDeviceInterfaces.restype = wintypes.BOOL
    _setupapi.SetupDiGetDeviceInterfaceDetailW.restype = wintypes.BOOL
    _setupapi.SetupDiDestroyDeviceInfoList.restype = wintypes.BOOL
    _kernel32.CreateFileW.restype = ctypes.c_void_p
    _kernel32.DeviceIoControl.restype = wintypes.BOOL
    _kernel32.CloseHandle.restype = wintypes.BOOL


# -----------------------------------------------------------------------
# Exception classes
# -----------------------------------------------------------------------


class ZeusEMINotSupportedError(ZeusBaseCPUError):
    """Raised when EMI is not available on this system."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class ZeusEMIInitError(ZeusBaseCPUError):
    """Raised when an EMI device cannot be opened or queried."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


# -----------------------------------------------------------------------
# Internal helpers (Windows-only)
# -----------------------------------------------------------------------


def _build_emi_guid() -> "_Guid":
    """Build the EMI device interface GUID structure."""
    guid = _Guid()
    guid.Data1 = _EMI_GUID_DATA1
    guid.Data2 = _EMI_GUID_DATA2
    guid.Data3 = _EMI_GUID_DATA3
    for i, byte in enumerate(_EMI_GUID_DATA4):
        guid.Data4[i] = byte
    return guid


def _get_emi_device_paths() -> list[str]:
    """Enumerate all EMI-compliant device interface paths on this Windows system.

    Returns an empty list if no devices are found or if not running on Windows.
    """
    guid = _build_emi_guid()

    hdev = _setupapi.SetupDiGetClassDevsW(
        ctypes.byref(guid),
        None,
        None,
        _DIGCF_PRESENT | _DIGCF_DEVICEINTERFACE,
    )
    if hdev == _INVALID_HANDLE_VALUE:
        logger.debug("SetupDiGetClassDevsW returned INVALID_HANDLE_VALUE for EMI GUID.")
        return []

    paths: list[str] = []
    try:
        index = 0
        while True:
            iface = _SpDeviceInterfaceData()
            iface.cbSize = ctypes.sizeof(_SpDeviceInterfaceData)
            if not _setupapi.SetupDiEnumDeviceInterfaces(
                ctypes.c_void_p(hdev),
                None,
                ctypes.byref(guid),
                index,
                ctypes.byref(iface),
            ):
                break

            # First call: obtain required buffer size.
            required = wintypes.DWORD(0)
            _setupapi.SetupDiGetDeviceInterfaceDetailW(
                ctypes.c_void_p(hdev),
                ctypes.byref(iface),
                None,
                0,
                ctypes.byref(required),
                None,
            )
            if required.value == 0:
                index += 1
                continue

            # Second call: fill the detail buffer.
            # SP_DEVICE_INTERFACE_DETAIL_DATA_W layout:
            #   DWORD cbSize  (4 bytes)
            #   WCHAR DevicePath[ANYSIZE_ARRAY]  (variable)
            detail_buf = ctypes.create_string_buffer(required.value)
            ctypes.cast(detail_buf, ctypes.POINTER(wintypes.DWORD))[0] = _DETAIL_DATA_CBSIZE
            if _setupapi.SetupDiGetDeviceInterfaceDetailW(
                ctypes.c_void_p(hdev),
                ctypes.byref(iface),
                detail_buf,
                required,
                None,
                None,
            ):
                # DevicePath starts immediately after the DWORD cbSize field.
                path = ctypes.wstring_at(ctypes.addressof(detail_buf) + 4)
                paths.append(path)
                logger.debug("Found EMI device: %s", path)

            index += 1
    finally:
        _setupapi.SetupDiDestroyDeviceInfoList(ctypes.c_void_p(hdev))

    return paths


def _ioctl(handle: "ctypes.c_void_p", code: int, out_size: int) -> bytes | None:
    """Send a buffered IOCTL with no input buffer and return the output bytes.

    Returns ``None`` if the call fails.
    """
    buf = ctypes.create_string_buffer(out_size)
    bytes_returned = wintypes.DWORD(0)
    ok = _kernel32.DeviceIoControl(
        handle,
        code,
        None,
        0,
        buf,
        out_size,
        ctypes.byref(bytes_returned),
        None,
    )
    if not ok:
        return None
    return bytes(buf)[: bytes_returned.value]


# -----------------------------------------------------------------------
# Channel metadata dataclass (plain class to keep it lightweight)
# -----------------------------------------------------------------------


class _EMIChannel:
    """Metadata for a single EMI channel."""

    __slots__ = ("index", "name", "unit")

    def __init__(self, index: int, name: str, unit: int) -> None:
        self.index = index
        self.name = name
        self.unit = unit


# -----------------------------------------------------------------------
# EMIFile — manages one open EMI device handle
# -----------------------------------------------------------------------


class EMIFile:
    """Manages an open Windows EMI device handle and reads energy data from it.

    Each ``EMIFile`` corresponds to one EMI device interface (one device path).
    It reads energy values in picowatt-hours and converts them to millijoules.

    Attributes:
        path (str): The Windows device interface path.
        version (int): The EMI interface version reported by the device (1 or 2).
        channels (list[_EMIChannel]): Metadata for each channel on this device.
    """

    def __init__(self, path: str) -> None:
        r"""Open the EMI device and read its metadata.

        Args:
            path: Windows device interface path (e.g. ``\\?\acpi#...``).

        Raises:
            ZeusEMIInitError: If the device cannot be opened or its metadata cannot be read.
        """
        self.path = path

        handle = _kernel32.CreateFileW(
            path,
            _GENERIC_READ,
            _FILE_SHARE_READ,
            None,
            _OPEN_EXISTING,
            0,
            None,
        )
        if handle == _INVALID_HANDLE_VALUE:
            err = _kernel32.GetLastError()
            raise ZeusEMIInitError(f"Failed to open EMI device '{path}' (Windows error {err}).")
        # Store the raw integer handle value so that __del__ can safely check
        # ``isinstance(self._handle, int)`` and avoid calling CloseHandle on
        # mocked handles during testing.
        self._handle: int = int(handle)

        try:
            self.version, self.channels = self._read_metadata()
        except ZeusEMIInitError:
            _kernel32.CloseHandle(self._handle)
            raise

    def _read_metadata(self) -> tuple[int, list[_EMIChannel]]:
        """Query the device for its version and channel metadata.

        Returns:
            A (version, channels) tuple.

        Raises:
            ZeusEMIInitError: On any IOCTL failure.
        """
        # Version
        ver_bytes = _ioctl(ctypes.c_void_p(self._handle), _IOCTL_EMI_GET_VERSION, 2)
        if ver_bytes is None or len(ver_bytes) < 2:
            raise ZeusEMIInitError(f"IOCTL_EMI_GET_VERSION failed for '{self.path}'.")
        version = int.from_bytes(ver_bytes[:2], "little")
        if version not in (_EMI_VERSION_V1, _EMI_VERSION_V2):
            raise ZeusEMIInitError(f"Unsupported EMI version {version} for '{self.path}'.")

        # Metadata size
        ms_bytes = _ioctl(ctypes.c_void_p(self._handle), _IOCTL_EMI_GET_METADATA_SIZE, 4)
        if ms_bytes is None or len(ms_bytes) < 4:
            raise ZeusEMIInitError(f"IOCTL_EMI_GET_METADATA_SIZE failed for '{self.path}'.")
        meta_size = int.from_bytes(ms_bytes[:4], "little")
        if meta_size == 0:
            raise ZeusEMIInitError(f"EMI device '{self.path}' reported zero metadata size.")

        # Metadata payload
        meta_bytes = _ioctl(ctypes.c_void_p(self._handle), _IOCTL_EMI_GET_METADATA, meta_size)
        if meta_bytes is None:
            raise ZeusEMIInitError(f"IOCTL_EMI_GET_METADATA failed for '{self.path}'.")

        channels = self._parse_channels(version, meta_bytes)
        return version, channels

    @staticmethod
    def _parse_channels(version: int, raw: bytes) -> list[_EMIChannel]:
        """Parse channel metadata from the raw metadata buffer.

        EMI_METADATA_V1 layout (offsets in bytes):
            MeasurementUnit  UINT   4
            HardwareOEM      WCHAR[16]  32
            HardwareModel    WCHAR[16]  32
            HardwareRevision USHORT 2
            MeteredHardwareNameSize USHORT 2
            MeteredHardwareName WCHAR[] variable

        EMI_METADATA_V2 layout (offsets in bytes):
            HardwareOEM      WCHAR[16]  32   @ 0
            HardwareModel    WCHAR[16]  32   @ 32
            HardwareRevision USHORT     2    @ 64
            ChannelCount     USHORT     2    @ 66
            Channels[]                       @ 68

        Each EMI_CHANNEL_V2:
            MeasurementUnit  UINT   4
            ChannelNameSize  USHORT 2   (bytes, including null terminator)
            ChannelName      WCHAR[] ChannelNameSize bytes
        """
        channels: list[_EMIChannel] = []

        if version == _EMI_VERSION_V2:
            # _EMI_NAME_MAX = 16 WCHARs = 32 bytes each for OEM and Model.
            _oem_size = _EMI_NAME_MAX * 2
            _model_size = _EMI_NAME_MAX * 2
            channel_count = int.from_bytes(
                raw[_oem_size + _model_size + 2 : _oem_size + _model_size + 4],
                "little",
            )
            offset = _oem_size + _model_size + 4  # skip OEM, Model, Revision, ChannelCount

            for i in range(channel_count):
                unit = int.from_bytes(raw[offset : offset + 4], "little")
                name_size = int.from_bytes(raw[offset + 4 : offset + 6], "little")
                name = (
                    raw[offset + 6 : offset + 6 + name_size].decode("utf-16-le").rstrip("\x00") if name_size > 0 else ""
                )
                if unit != _EMI_MEASUREMENT_UNIT_PICOWATT_HOURS:
                    logger.warning(
                        "EMI channel %d ('%s') uses unexpected unit %d; expected picowatt-hours (0).",
                        i,
                        name,
                        unit,
                    )
                channels.append(_EMIChannel(index=i, name=name, unit=unit))
                offset += 4 + 2 + name_size

        elif version == _EMI_VERSION_V1:
            # Single channel: the whole device is one domain.
            unit = int.from_bytes(raw[0:4], "little")
            _oem_size = _EMI_NAME_MAX * 2
            _model_size = _EMI_NAME_MAX * 2
            name_size = int.from_bytes(
                raw[4 + _oem_size + _model_size + 2 : 4 + _oem_size + _model_size + 4],
                "little",
            )
            name = (
                raw[4 + _oem_size + _model_size + 4 : 4 + _oem_size + _model_size + 4 + name_size]
                .decode("utf-16-le")
                .rstrip("\x00")
                if name_size > 0
                else "EMI_V1"
            )
            if unit != _EMI_MEASUREMENT_UNIT_PICOWATT_HOURS:
                logger.warning(
                    "EMI V1 channel ('%s') uses unexpected unit %d; expected picowatt-hours (0).",
                    name,
                    unit,
                )
            channels.append(_EMIChannel(index=0, name=name, unit=unit))

        return channels

    def read(self, channel_index: int) -> float:
        """Read the accumulated energy for the given channel.

        Args:
            channel_index: Zero-based index of the channel within this device.

        Returns:
            The accumulated energy in millijoules.

        Raises:
            ZeusEMIInitError: If the IOCTL call fails.
        """
        meas_size = len(self.channels) * 16  # 16 bytes per EMI_CHANNEL_MEASUREMENT_DATA
        meas_bytes = _ioctl(ctypes.c_void_p(self._handle), _IOCTL_EMI_GET_MEASUREMENT, meas_size)
        if meas_bytes is None:
            err = _kernel32.GetLastError()
            raise ZeusEMIInitError(f"IOCTL_EMI_GET_MEASUREMENT failed for '{self.path}' (Windows error {err}).")
        # EMI_MEASUREMENT_DATA_V2: ChannelData[ChannelCount]
        # Each EMI_CHANNEL_MEASUREMENT_DATA: AbsoluteEnergy (8 bytes) + AbsoluteTime (8 bytes)
        offset = channel_index * 16
        energy_pwh = int.from_bytes(meas_bytes[offset : offset + 8], "little")
        return energy_pwh * _PICOWATT_HOURS_TO_MILLIJOULES

    def __del__(self) -> None:
        """Close the device handle."""
        if not _WINDOWS:
            return
        handle = getattr(self, "_handle", 0)
        if isinstance(handle, int) and handle and handle != _INVALID_HANDLE_VALUE:
            _kernel32.CloseHandle(ctypes.c_void_p(handle))
            self._handle = 0


# -----------------------------------------------------------------------
# EMICPU — one CPU package
# -----------------------------------------------------------------------


class EMICPU(cpu_common.CPU):
    """Reads energy for a single Intel CPU package via the Windows EMI interface.

    Attributes:
        cpu_index (int): Zero-based package (socket) index.
    """

    def __init__(
        self,
        cpu_index: int,
        emi_file: EMIFile,
        pkg_channel_index: int,
        dram_channel_index: int | None,
    ) -> None:
        """Initialize the EMICPU.

        Args:
            cpu_index: Zero-based CPU package (socket) index.
            emi_file: The :class:`EMIFile` that owns the device handle.
            pkg_channel_index: Index of the PKG (package) energy channel in ``emi_file``.
            dram_channel_index: Index of the DRAM energy channel, or ``None`` if unavailable.
        """
        super().__init__(cpu_index)
        self._emi_file = emi_file
        self._pkg_channel_index = pkg_channel_index
        self._dram_channel_index = dram_channel_index

    def get_total_energy_consumption(self) -> CpuDramMeasurement:
        """Return the total accumulated energy for this CPU package. Units: mJ."""
        cpu_mj = self._emi_file.read(self._pkg_channel_index)
        dram_mj: float | None = None
        if self._dram_channel_index is not None:
            dram_mj = self._emi_file.read(self._dram_channel_index)
        return CpuDramMeasurement(cpu_mj=cpu_mj, dram_mj=dram_mj)

    def supports_get_dram_energy_consumption(self) -> bool:
        """Return ``True`` if DRAM energy data is available for this package."""
        return self._dram_channel_index is not None


# -----------------------------------------------------------------------
# EMICPUs — manager for all packages
# -----------------------------------------------------------------------


class EMICPUs(cpu_common.CPUs):
    """Manages all Intel CPU packages accessible via the Windows EMI interface.

    Each detected ``RAPL_Package{N}_PKG`` EMI channel maps to one
    :class:`EMICPU` object at index N.
    """

    def __init__(self) -> None:
        """Discover and initialise all EMI CPU objects.

        Raises:
            ZeusEMINotSupportedError: If EMI is unavailable on this system.
            ZeusEMIInitError: If device metadata cannot be queried.
        """
        if not emi_is_available():
            raise ZeusEMINotSupportedError("No EMI-compatible energy meter devices were found on this system.")
        self._cpus: list[EMICPU] = []
        self._init_cpus()

    def _init_cpus(self) -> None:
        """Build the list of :class:`EMICPU` objects from all EMI devices."""
        paths = _get_emi_device_paths()

        # pkg_index → (EMIFile, pkg_channel_idx, dram_channel_idx | None)
        packages: dict[int, tuple[EMIFile, int, int | None]] = {}

        # Pattern: RAPL_Package{N}_PKG  or  RAPL_Package{N}_DRAM
        _pkg_re = re.compile(r"RAPL_Package(\d+)_PKG$", re.IGNORECASE)
        _dram_re = re.compile(r"RAPL_Package(\d+)_DRAM$", re.IGNORECASE)

        for path in paths:
            try:
                emi_file = EMIFile(path)
            except ZeusEMIInitError as err:
                logger.warning("Skipping EMI device '%s': %s", path, err)
                continue

            # Locate PKG and DRAM channels for each package on this device.
            pkg_channels: dict[int, int] = {}
            dram_channels: dict[int, int] = {}
            for ch in emi_file.channels:
                m = _pkg_re.match(ch.name)
                if m:
                    pkg_channels[int(m.group(1))] = ch.index
                    continue
                m = _dram_re.match(ch.name)
                if m:
                    dram_channels[int(m.group(1))] = ch.index

            for pkg_num, pkg_idx in pkg_channels.items():
                dram_idx = dram_channels.get(pkg_num)
                packages[pkg_num] = (emi_file, pkg_idx, dram_idx)

        if not packages:
            raise ZeusEMINotSupportedError("EMI devices were found but no RAPL_Package PKG channels were detected.")

        for pkg_num in sorted(packages):
            emi_file, pkg_idx, dram_idx = packages[pkg_num]
            self._cpus.append(
                EMICPU(
                    cpu_index=pkg_num,
                    emi_file=emi_file,
                    pkg_channel_index=pkg_idx,
                    dram_channel_index=dram_idx,
                )
            )
            logger.info(
                "Initialized EMI CPU %d: PKG channel=%d, DRAM channel=%s",
                pkg_num,
                pkg_idx,
                dram_idx,
            )

    @property
    def cpus(self) -> Sequence[EMICPU]:
        """Return the list of :class:`EMICPU` objects."""
        return self._cpus

    def __del__(self) -> None:
        """Clean up resources."""
        pass


# -----------------------------------------------------------------------
# Availability check
# -----------------------------------------------------------------------


@lru_cache(maxsize=1)
def emi_is_available() -> bool:
    """Return ``True`` if at least one EMI energy meter device is present.

    This function is cached — the check runs at most once per process.
    On non-Windows platforms it always returns ``False``.
    """
    if not _WINDOWS:
        logger.info("EMI is not supported on non-Windows platforms.")
        return False
    try:
        paths = _get_emi_device_paths()
    except Exception as err:
        logger.info("EMI device enumeration failed: %s", err)
        return False
    if paths:
        logger.info(
            "EMI is available: found %d device(s).",
            len(paths),
        )
        return True
    logger.info("No EMI energy meter devices found.")
    return False
