"""Raspberry Pi 5 whole-machine power via the on-board PMIC.

The Pi 5 is the first Pi whose PMIC exposes per-rail current and voltage to
userspace. Twelve rails carry both channels on a Model B Rev 1.0, including the
SoC core rail and the two DRAM rails, so the board can be metered as a whole and
broken down at the same time. `EXT5V` and `BATT` report a voltage and no
current, so they cannot yield power and are skipped.

The PMIC is a separate chip metering board rails rather than an on-die sensor,
which is why this lives under `machine` and not under `soc`.

There is no cumulative energy counter in the hardware, so power is sampled and
integrated here, the same shape the Jetson SoC backend uses.
"""

from __future__ import annotations

import abc
import array
import asyncio
import atexit
import contextlib
import enum
import multiprocessing as mp
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty

try:
    import fcntl
except ImportError:  # pragma: no cover - the mailbox is a Linux device
    fcntl = None  # ty: ignore[invalid-assignment]

from zeus.device.machine.common import (
    Machine,
    MachineMeasurement,
    ZeusMachineInitError,
)

# `vcgencmd pmic_read_adc` prints one line per ADC channel, e.g.
#
#      VDD_CORE_A current(7)=0.81262990A
#      VDD_CORE_V volt(15)=0.86065850V
#        EXT5V_V volt(24)=5.07860000V
#
# Channels ending in `_A` carry current and channels ending in `_V` carry
# voltage; the shared prefix is the rail name.
_ADC_LINE = re.compile(r"^\s*(?P<rail>\S+?)_(?P<kind>[AV])\s+(?:current|volt)\(\d+\)=(?P<value>[\d.]+)[AV]\s*$")

_CPU_RAILS = ("VDD_CORE",)

# Both DRAM rails are read. Characterised on a Pi 5 Model B Rev 1.0 at about
# 50 Hz, 150 samples per condition: under a decode workload DDR_VDD2 is non-zero
# in 148 of 150 samples (mean 0.212 W) and DDR_VDDQ in 135 of 150 (mean
# 0.034 W), so VDDQ carries roughly 14% of DRAM power rather than nothing. At
# idle both read exact zero in essentially every sample, which is a property of
# the ADC at low current rather than of the rail, and it means a low-rate
# sampler will underestimate DRAM energy on a lightly loaded board.
_DRAM_RAILS = ("DDR_VDD2", "DDR_VDDQ")

# 10 Hz. Sampling costs about 23 ms of firmware ADC time per read on this board,
# almost all of it inside the firmware rather than in the caller, so a faster
# poll takes measurable CPU away from the workload being measured.
_DEFAULT_POLL_INTERVAL_S = 0.1

_VCIO = "/dev/vcio"
_IOCTL_MBOX_PROPERTY = 0xC0086400  # _IOWR(100, 0, 8)
_TAG_VCGENCMD_STRING = 0x00030080
_PAYLOAD_OFFSET = 24  # bytes; header is six 32-bit words
_BUFBYTES = 1024


class ZeusRaspberryPiInitError(ZeusMachineInitError):
    """Raspberry Pi initialization failures."""

    def __init__(self, message: str) -> None:
        """Initialize Zeus Exception."""
        super().__init__(message)


class PMICReadError(RuntimeError):
    """Raised when the PMIC ADC cannot be read."""


class PMICReader(abc.ABC):
    """Reads instantaneous per-rail power from the board's PMIC."""

    @abc.abstractmethod
    def read_rail_power_mw(self) -> dict[str, float]:
        """Return a mapping of rail name to instantaneous power in mW."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release any resources held. Readers that hold none may do nothing."""

    @staticmethod
    def parse(output: str) -> dict[str, float]:
        """Parse `pmic_read_adc` output into a rail name to power (mW) mapping."""
        amps: dict[str, float] = {}
        volts: dict[str, float] = {}
        for line in output.replace("\x00", "\n").splitlines():
            match = _ADC_LINE.match(line)
            if match is None:
                continue
            rail = match.group("rail")
            value = float(match.group("value"))
            if match.group("kind") == "A":
                amps[rail] = value
            else:
                volts[rail] = value

        # Only rails carrying both channels yield power.
        return {rail: amps[rail] * volts[rail] * 1000.0 for rail in amps.keys() & volts.keys()}


class VcioPMICReader(PMICReader):
    """Reads the PMIC through the firmware mailbox on a persistent descriptor.

    `vcgencmd pmic_read_adc` does two things, which strace confirms: it opens
    `/dev/vcio` and issues one `_IOWR(100, 0, 8)` mailbox property ioctl carrying
    the command string under firmware tag 0x00030080. Doing that directly avoids
    a process spawn per sample.

    Worth being honest about the size of the win. Measured over 50 reads on a
    Pi 5, this costs 22.6 ms per sample against 23.9 ms for the subprocess, so it
    saves about 5%. The firmware's own ADC sweep dominates, and no userspace
    change touches that. The reason to prefer it is that it does not fork a
    process into the middle of the workload being measured, not that it is fast.

    `/dev/vcio` is root:video mode 0660, so membership of the video group is
    enough and no root is required.
    """

    def __init__(self, path: str = _VCIO) -> None:
        """Open the mailbox device."""
        try:
            self._fd: int | None = os.open(path, os.O_RDONLY)
        except OSError as exc:
            raise PMICReadError(
                f"Could not open {path}: {exc}. The mailbox device is root:video "
                "mode 0660, so this usually means the user is not in the video group."
            ) from exc
        self.path = path

    def close(self) -> None:
        """Close the mailbox descriptor."""
        if self._fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = None

    def __enter__(self) -> VcioPMICReader:
        """Return self for use as a context manager."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the descriptor on exit."""
        self.close()

    def _command(self, command: str) -> str:
        if fcntl is None:
            raise PMICReadError("The firmware mailbox needs fcntl, which is Linux only.")
        if self._fd is None:
            raise PMICReadError("Mailbox descriptor is closed.")
        words = _BUFBYTES // 4
        buf = array.array("I", [0] * words)
        buf[0] = _BUFBYTES
        buf[1] = 0
        buf[2] = _TAG_VCGENCMD_STRING
        buf[3] = _BUFBYTES - _PAYLOAD_OFFSET
        buf[4] = 0
        raw = memoryview(buf).cast("B")
        payload = command.encode() + b"\0"
        raw[_PAYLOAD_OFFSET : _PAYLOAD_OFFSET + len(payload)] = payload
        buf[words - 1] = 0

        try:
            # ty checks every platform (python-platform = "all"), and fcntl is
            # Linux only. The None guard above covers the platforms it is missing on.
            fcntl.ioctl(self._fd, _IOCTL_MBOX_PROPERTY, buf, True)  # ty: ignore[possibly-missing-attribute]
        except OSError as exc:
            raise PMICReadError(f"Mailbox ioctl failed: {exc}") from exc
        if buf[1] != 0x80000000:
            raise PMICReadError(f"Mailbox returned status 0x{buf[1]:08x}")

        blob = memoryview(buf).cast("B")[_PAYLOAD_OFFSET:].tobytes()
        return blob.split(b"\x00\x00")[0].decode("utf-8", "replace")

    def read_rail_power_mw(self) -> dict[str, float]:
        """Return a mapping of rail name to instantaneous power in mW."""
        text = self._command("pmic_read_adc")
        # The firmware answers an unregistered command with a body that parses
        # to nothing. Catching it here keeps a rejected command from looking
        # like a board with no rails.
        if "error_msg" in text:
            raise PMICReadError(f"Firmware rejected pmic_read_adc: {text.strip()}")
        return self.parse(text)


class VcgencmdPMICReader(PMICReader):
    """Reads the PMIC by running `vcgencmd pmic_read_adc`.

    Kept as the fallback for hosts where `/dev/vcio` is not reachable, for
    example inside a container that does not map the device.
    """

    def __init__(self, vcgencmd: str = "vcgencmd", timeout_s: float = 5.0) -> None:
        """Initialize the reader with the path to `vcgencmd`."""
        self.vcgencmd = vcgencmd
        self.timeout_s = timeout_s

    def _run(self) -> str:
        try:
            proc = subprocess.run(
                [self.vcgencmd, "pmic_read_adc"],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise PMICReadError(f"Could not run `{self.vcgencmd} pmic_read_adc`: {exc}") from exc
        if proc.returncode != 0:
            raise PMICReadError(f"`{self.vcgencmd} pmic_read_adc` exited {proc.returncode}: {proc.stderr.strip()}")
        return proc.stdout

    def read_rail_power_mw(self) -> dict[str, float]:
        """Return a mapping of rail name to instantaneous power in mW."""
        return self.parse(self._run())

    def close(self) -> None:
        """No resources are held between reads."""


def default_reader() -> PMICReader:
    """Return the mailbox reader when it is usable, otherwise the subprocess one."""
    try:
        reader = VcioPMICReader()
    except PMICReadError:
        return VcgencmdPMICReader()
    try:
        if reader.read_rail_power_mw():
            return reader
    except PMICReadError:
        pass
    reader.close()
    return VcgencmdPMICReader()


@dataclass
class RPIMeasurement(MachineMeasurement):
    """Energy measured at the Raspberry Pi 5 PMIC. Units: mJ.

    Attributes:
        machine_energy_mj: Summed across every rail that reports both a current
            and a voltage, twelve on a Model B Rev 1.0. Whole-board, so it
            includes the wireless, HDMI and I/O rails and is larger than the sum
            of the two fields below.
        cpu_energy_mj: The SoC core rail, `VDD_CORE`.
        dram_energy_mj: The two DRAM rails, `DDR_VDD2` and `DDR_VDDQ`, combined.
    """

    cpu_energy_mj: float | None = None
    dram_energy_mj: float | None = None


class Command(enum.Enum):
    """Commands accepted by the polling process."""

    READ = "read"
    STOP = "stop"


class RaspberryPi(Machine):
    """Whole-machine energy for a Raspberry Pi 5, measured at the PMIC."""

    def __init__(
        self,
        reader: PMICReader | None = None,
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        """Initialize a Raspberry Pi energy monitor.

        Args:
            reader: PMIC reader to use. Defaults to the mailbox reader, falling
                back to `vcgencmd` when `/dev/vcio` is not usable.
            poll_interval_s: Seconds between PMIC samples.
        """
        self.reader = reader if reader is not None else default_reader()
        self.poll_interval_s = poll_interval_s
        self.measurement_states: dict[str, RPIMeasurement] = {}
        self.available_metrics: set[str] | None = None

        # Fail with a clear message rather than starting a polling process that
        # can only ever produce zeros.
        try:
            rails = self.reader.read_rail_power_mw()
        except PMICReadError as exc:
            raise ZeusRaspberryPiInitError(str(exc)) from exc
        if not rails:
            raise ZeusRaspberryPiInitError(
                "The PMIC ADC reported no rail with both a current and a voltage "
                "channel. Per-rail energy measurement needs a board whose PMIC "
                "exposes both."
            )
        self.rails = sorted(rails)

        self.command_queue: mp.Queue = mp.Queue()
        self.result_queue: mp.Queue = mp.Queue()
        self.process = mp.Process(
            target=_polling_process_async_wrapper,
            args=(
                self.command_queue,
                self.result_queue,
                self.reader,
                self.poll_interval_s,
            ),
            daemon=True,
        )
        self.process.start()
        atexit.register(self._stop_process)

    def _stop_process(self) -> None:
        """Stop the polling process."""
        with contextlib.suppress(Exception):
            self.command_queue.put_nowait(Command.STOP)
        self.process.join(timeout=1.0)
        if self.process.is_alive():
            self.process.kill()

    def get_available_metrics(self) -> set[str]:
        """Return the measurement fields this board actually populates."""
        if self.available_metrics is None:
            result = self.get_total_energy_consumption()
            self.available_metrics = {name for name, value in asdict(result).items() if value is not None}
        return self.available_metrics

    def get_total_energy_consumption(self, timeout: float = 15.0) -> RPIMeasurement:
        """Return cumulative board energy since the monitor started. Units: mJ."""
        self.command_queue.put(Command.READ)
        return self.result_queue.get(timeout=timeout)

    def begin_window(self, key: str, restart: bool = False) -> None:
        """Begin a measurement interval labeled with `key`.

        Args:
            key: Unique name of the measurement window.
            restart: If True and the window already exists, cancel the existing
                window and start a new one.
        """
        if key in self.measurement_states:
            if not restart:
                raise KeyError(f"Measurement window '{key}' already exists")
            self.measurement_states.pop(key)
        self.measurement_states[key] = self.get_total_energy_consumption()

    def end_window(self, key: str) -> RPIMeasurement:
        """End a measurement window and return the energy consumed. Units: mJ."""
        try:
            start = self.measurement_states.pop(key)
        except KeyError:
            raise KeyError(f"Measurement window '{key}' does not exist") from None
        return self.get_total_energy_consumption() - start


def _accumulate(
    cumulative: RPIMeasurement,
    prev_power: dict[str, float],
    power: dict[str, float],
    dt: float,
) -> None:
    """Trapezoidally integrate one sampling interval into `cumulative`.

    Trapezoidal rather than rectangular because at 10 Hz the power on this board
    can change substantially within one interval, a core leaving idle moves the
    core rail by several watts, and taking either endpoint alone biases the
    integral in a load-dependent direction.
    """

    def rail_energy(rail_names: tuple[str, ...]) -> float | None:
        total = 0.0
        seen = False
        for rail in rail_names:
            if rail in power and rail in prev_power:
                total += 0.5 * (power[rail] + prev_power[rail]) * dt
                seen = True
        return total if seen else None

    cpu = rail_energy(_CPU_RAILS)
    if cpu is not None:
        cumulative.cpu_energy_mj = (cumulative.cpu_energy_mj or 0.0) + cpu

    dram = rail_energy(_DRAM_RAILS)
    if dram is not None:
        cumulative.dram_energy_mj = (cumulative.dram_energy_mj or 0.0) + dram

    shared = power.keys() & prev_power.keys()
    if shared:
        total = sum(0.5 * (power[r] + prev_power[r]) * dt for r in shared)
        cumulative.machine_energy_mj += total


def _polling_process_async_wrapper(
    command_queue: mp.Queue,
    result_queue: mp.Queue,
    reader: PMICReader,
    poll_interval_s: float,
) -> None:
    """Function wrapper for the asynchronous energy polling process."""
    asyncio.run(_polling_process_async(command_queue, result_queue, reader, poll_interval_s))


async def _polling_process_async(
    command_queue: mp.Queue,
    result_queue: mp.Queue,
    reader: PMICReader,
    poll_interval_s: float,
) -> None:
    """Continuously integrate PMIC power into cumulative energy until told to stop."""
    probe = reader.read_rail_power_mw()
    cumulative = RPIMeasurement(
        machine_energy_mj=0.0,
        cpu_energy_mj=0.0 if any(r in probe for r in _CPU_RAILS) else None,
        dram_energy_mj=0.0 if any(r in probe for r in _DRAM_RAILS) else None,
    )

    prev_power = probe
    prev_ts = time.monotonic()

    while True:
        try:
            power = reader.read_rail_power_mw()
        except PMICReadError:
            # A transient read failure should not lose the accumulated total;
            # reuse the previous sample for this interval and try again.
            power = prev_power
        now = time.monotonic()
        _accumulate(cumulative, prev_power, power, now - prev_ts)
        prev_power = power
        prev_ts = now

        try:
            command = await asyncio.to_thread(command_queue.get, timeout=poll_interval_s)
        except Empty:
            continue

        if command == Command.STOP:
            break
        if command == Command.READ:
            result_queue.put(cumulative)


def rpi_is_available() -> bool:
    """Return whether this is a Raspberry Pi whose PMIC ADC can actually be read.

    Boards before the Pi 5 do not expose per-rail ADC channels, so the model
    string alone is not enough and the read has to return rails.
    """
    if sys.platform != "linux" or platform.machine() not in ("aarch64", "arm64"):
        return False

    model = Path("/proc/device-tree/model")
    try:
        if "raspberry pi" not in model.read_bytes().decode(errors="ignore").lower():
            return False
    except OSError:
        return False

    reader: PMICReader | None = None
    try:
        reader = default_reader()
        return bool(reader.read_rail_power_mw())
    except PMICReadError:
        return False
    finally:
        if reader is not None:
            reader.close()
