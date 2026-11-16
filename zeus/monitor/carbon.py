"""Carbon intensity providers used for carbon-aware optimizers."""

from __future__ import annotations

import abc
import importlib.util
import json
import math
import queue
import logging
import requests
import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal
from datetime import datetime, timezone, timedelta
from dateutil import parser
from collections import defaultdict
from collections.abc import Mapping

import pandas as pd

from zeus.exception import ZeusBaseError
from zeus.monitor import ZeusMonitor
from zeus.utils.multiprocessing import warn_if_global_in_subprocess

logger = logging.getLogger(__name__)


class ZeusCarbonIntensityHTTPError(ZeusBaseError):
    """Exception when HTTP request to carbon intensity provider fails."""

    def __init__(self, message: str) -> None:
        """Initialize HTTP request exception."""
        super().__init__(message)


class ZeusCarbonIntensityNotFoundError(ZeusBaseError):
    """Exception when carbon intensity measurement could not be retrieved."""

    def __init__(self, message: str) -> None:
        """Initialize carbon not found exception."""
        super().__init__(message)


class CarbonIntensityProvider(abc.ABC):
    """Abstract class for implementing ways to fetch carbon intensity."""

    @abc.abstractmethod
    def get_current_carbon_intensity(self) -> float:
        """Abstract method for fetching the current carbon intensity of the set location of the class."""
        pass

    @abc.abstractmethod
    def get_recent_carbon_intensity(self) -> dict[datetime, float]:
        """Abstract method for fetching the current carbon intensity of the set location of the class."""
        pass

    @property
    @abc.abstractmethod
    def update_period(self) -> timedelta:
        """Abstract method for how long each carbon intensity value in the history dict remains current."""
        pass

    @property
    @abc.abstractmethod
    def history_length(self) -> int:
        """Abstract method for how many carbon intensity values are in the history dict."""
        pass


class ElectrictyMapsClient(CarbonIntensityProvider):
    """Carbon Intensity Provider with ElectricityMaps API.

    Reference:

    1. [ElectricityMaps](https://www.electricitymaps.com/)
    2. [ElectricityMaps API](https://static.electricitymaps.com/api/docs/index.html)
    3. [ElectricityMaps GitHub](https://github.com/electricitymaps/electricitymaps-contrib)
    """

    def __init__(
        self,
        location: tuple[float, float],
        estimate: bool = False,
        emission_factor_type: Literal["direct", "lifecycle"] = "direct",
    ) -> None:
        """Iniitializes ElectricityMaps Carbon Provider.

        Args:
            location: tuple of latitude and longitude (latitude, longitude)
            estimate: bool to toggle whether carbon intensity is estimated or not
            emission_factor_type: emission factor to be measured (`direct` or `lifestyle`)
        """
        self.lat, self.long = location
        self.estimate = estimate
        self.emission_factor_type = emission_factor_type

    def get_current_carbon_intensity(self) -> float:
        """Fetches current carbon intensity of the location of the class.

        In some locations, there is no recent carbon intensity data. `self.estimate` can be used to approximate the carbon intensity in such cases.
        """
        try:
            url = (
                f"https://api.electricitymap.org/v3/carbon-intensity/latest?lat={self.lat}&lon={self.long}"
                + f"&disableEstimations={not self.estimate}&emissionFactorType={self.emission_factor_type}"
            )
            resp = requests.get(url)
        except requests.exceptions.RequestException as e:
            raise ZeusCarbonIntensityHTTPError(f"Failed to retrieve current carbon intensity measurement: {e}") from e

        try:
            return resp.json()["carbonIntensity"]
        except KeyError as e:
            # Raise exception when carbonIntensity does not exist in response
            raise ZeusCarbonIntensityNotFoundError(
                f"Current carbon intensity measurement not found at `({self.lat}, {self.long})` "
                f"with estimate set to `{self.estimate}` and emission_factor_type set to `{self.emission_factor_type}`\n"
                f"JSON Response: {resp.text}"
            ) from e

    def get_recent_carbon_intensity(self) -> dict[datetime, float]:
        """Fetches recent (within last 24 hours) carbon intensity of the location of the class.

        In some locations, there is no recent carbon intensity data. `self.estimate` can be used to approximate the carbon intensity in such cases.
        """
        try:
            url = (
                f"https://api.electricitymap.org/v3/carbon-intensity/history?lat={self.lat}&lon={self.long}"
                + f"&disableEstimations={not self.estimate}&emissionFactorType={self.emission_factor_type}"
            )
            resp = requests.get(url)
        except requests.exceptions.RequestException as e:
            raise ZeusCarbonIntensityHTTPError(f"Failed to retrieve recent carbon intensity measurement: {e}") from e

        try:
            recent_carbon_intensities: dict[datetime, float] = {
                parser.parse(measurement["datetime"]): measurement["carbonIntensity"]
                for measurement in resp.json()["history"]
            }
            return recent_carbon_intensities
        except KeyError as e:
            # Raise exception when carbonIntensity does not exist in response
            raise ZeusCarbonIntensityNotFoundError(
                f"Recent carbon intensity measurement not found at `({self.lat}, {self.long})` "
                f"with estimate set to `{self.estimate}` and emission_factor_type set to `{self.emission_factor_type}`\n"
                f"JSON Response: {resp.text}"
            ) from e

    @property
    def update_period(self) -> timedelta:
        """Returns timedelta for how long each carbon intensity value in the history dict remains current."""
        return timedelta(hours=1)

    @property
    def history_length(self) -> int:
        """Returns number of carbon intensity values in history dict."""
        return 24


class TraceCarbonIntensityProvider(CarbonIntensityProvider):
    """Carbon intensity read from a recorded trace, with no network access.

    This makes carbon measurements reproducible and lets you ask what a job would
    have emitted on a past day.

    Each value covers one period of the trace: an hour for most grid data, or 30
    minutes for the GB Carbon Intensity API. The period is inferred from the data
    (the smallest gap between timestamps) unless given. Timestamps are period starts
    on a UTC grid of that length, for example the top of each UTC hour, which is how
    `CarbonEmissionMonitor` groups energy. A lookup returns the value of the period
    that contains the requested time (a step function, no interpolation). Periods
    missing from the trace raise `ZeusCarbonIntensityNotFoundError`. Nothing is
    filled in or extrapolated.

    Downloaded files can be loaded as they are with `from_electricitymaps_csv`,
    `from_eia930_xlsx` and `from_gb_carbon_intensity_json`, and other CSV files with
    `from_csv`.

    By default, the trace is read at the current wall-clock time. To replay a past
    day, pass `replay_start`: the period in which the provider is created is mapped
    to the period that contains `replay_start`, and later periods follow one to one.

    !!! Note
        Do not mix direct (combustion) and lifecycle emission factors when you compare
        results. EIA-930 CO2 estimates are direct emissions.
    """

    def __init__(
        self,
        intensities: Mapping[datetime, float],
        period: timedelta | None = None,
        replay_start: datetime | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            intensities: Maps the start of each period (timezone-aware) to the carbon
                intensity in gCO2eq/kWh. Values must be finite and non-negative.
            period: Length of each period. Inferred from the timestamps if omitted
                (one hour for a single timestamp).
            replay_start: A timezone-aware time inside the trace. If given, the current
                wall-clock period is mapped to the period that contains this time.
        """
        if not intensities:
            raise ValueError("The carbon intensity trace is empty.")
        trace: dict[datetime, float] = {}
        for start, value in intensities.items():
            if start.tzinfo is None:
                raise ValueError(f"Trace time {start} has no timezone. Use timezone-aware times.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Carbon intensity {value} at {start} is not a finite non-negative number.")
            trace[start.astimezone(timezone.utc)] = float(value)
        self.trace = dict(sorted(trace.items()))

        starts = list(self.trace)
        if period is None:
            period = min((b - a for a, b in zip(starts, starts[1:])), default=timedelta(hours=1))
        if period <= timedelta(0):
            raise ValueError("`period` must be positive.")
        self.period = period
        for start in starts:
            if _period_floor(start, period) != start:
                raise ValueError(
                    f"Trace time {start} does not start a {period} period on the UTC grid. "
                    "Timestamps must be period starts, for example the top of each hour for hourly data."
                )

        self.offset = timedelta(0)
        if replay_start is not None:
            if replay_start.tzinfo is None:
                raise ValueError("`replay_start` must be timezone-aware.")
            start = _period_floor(replay_start, period)
            if start not in self.trace:
                raise ValueError(f"`replay_start` {replay_start} is not inside the trace.")
            self.offset = start - _period_floor(datetime.now(timezone.utc), period)

    @classmethod
    def _from_series(
        cls,
        times: pd.Series,
        values: pd.Series,
        scale: float = 1.0,
        time_label: Literal["start", "end"] = "start",
        period: timedelta | None = None,
        replay_start: datetime | None = None,
    ) -> TraceCarbonIntensityProvider:
        """Build the provider from timezone-aware times and intensities, skipping empty values."""
        times = pd.Series(pd.to_datetime(times, utc=True)).reset_index(drop=True)
        values = pd.to_numeric(pd.Series(values), errors="coerce").reset_index(drop=True) * scale
        keep = values.notna()
        times, values = times[keep], values[keep]
        if times.duplicated().any():
            raise ValueError("The trace has repeated timestamps.")
        if time_label == "end":
            if period is None:
                diffs = times.sort_values().diff().dropna()
                period = diffs.min().to_pytimedelta() if len(diffs) else timedelta(hours=1)
            times = times - period
        trace = {t.to_pydatetime(): float(v) for t, v in zip(times, values)}
        return cls(trace, period=period, replay_start=replay_start)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        time_column: str,
        intensity_column: str,
        time_zone: str | None = None,
        time_label: Literal["start", "end"] = "start",
        scale: float = 1.0,
        period: timedelta | None = None,
        replay_start: datetime | None = None,
    ) -> TraceCarbonIntensityProvider:
        """Build the provider from any CSV file with a time column and an intensity column.

        Args:
            path: Path to the CSV file.
            time_column: Column with the timestamp of each period.
            intensity_column: Column with the carbon intensity of each period.
            time_zone: IANA time zone (for example `"UTC"` or `"America/Chicago"`) for
                timestamps that have no offset. Must be omitted if the timestamps carry
                an offset. Prefer UTC timestamps, since local times are ambiguous
                when daylight saving time ends.
            time_label: Whether each timestamp marks the `"start"` or the `"end"` of its period.
            scale: Factor that converts the column to gCO2eq/kWh. For example, 453.59237
                for lb/kWh, or 1000 for t/MWh.
            period: Passed to the constructor.
            replay_start: Passed to the constructor.
        """
        df = pd.read_csv(path)
        for column in (time_column, intensity_column):
            if column not in df.columns:
                raise ValueError(f"Column '{column}' is not in {path}.")
        text = df[time_column].astype(str).str.strip()
        has_offset = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$")
        if has_offset.all():
            if time_zone is not None:
                raise ValueError(f"Column '{time_column}' already has UTC offsets. Do not pass `time_zone`.")
            times = pd.to_datetime(text, utc=True)
        elif not has_offset.any():
            if time_zone is None:
                raise ValueError(f"Column '{time_column}' has no UTC offsets. Pass `time_zone`.")
            times = pd.to_datetime(text).dt.tz_localize(time_zone)
        else:
            raise ValueError(f"Column '{time_column}' mixes times with and without UTC offsets.")
        return cls._from_series(times, df[intensity_column], scale, time_label, period, replay_start)

    @classmethod
    def from_electricitymaps_csv(
        cls,
        path: str | Path,
        emission_factor_type: Literal["direct", "lifecycle"] = "lifecycle",
        replay_start: datetime | None = None,
    ) -> TraceCarbonIntensityProvider:
        """Load a CSV file downloaded from Electricity Maps, as it is.

        Reads `Datetime (UTC)` and the direct or lifecycle carbon intensity column. Both
        the 2023 headers (`Carbon Intensity gCO₂eq/kWh (LCA)`) and the 2024 headers
        (`Carbon intensity gCO₂eq/kWh (Life cycle)`) work. Use the hourly files; monthly
        and yearly files do not have regular periods.

        Args:
            path: Path to the CSV file.
            emission_factor_type: `"direct"` or `"lifecycle"`, as in `ElectrictyMapsClient`.
            replay_start: Passed to the constructor.
        """
        df = pd.read_csv(path)
        names = {
            "direct": ("Carbon intensity gCO₂eq/kWh (direct)", "Carbon Intensity gCO₂eq/kWh (direct)"),
            "lifecycle": ("Carbon intensity gCO₂eq/kWh (Life cycle)", "Carbon Intensity gCO₂eq/kWh (LCA)"),
        }[emission_factor_type]
        column = next((c for c in names if c in df.columns), None)
        if column is None or "Datetime (UTC)" not in df.columns:
            raise ValueError(
                f"{path} does not look like an Electricity Maps CSV: expected 'Datetime (UTC)' and one of {names}."
            )
        times = pd.to_datetime(df["Datetime (UTC)"]).dt.tz_localize("UTC")
        return cls._from_series(times, df[column], replay_start=replay_start)

    @classmethod
    def from_eia930_xlsx(
        cls,
        path: str | Path,
        electricity: Literal["consumed", "generated"] = "consumed",
        replay_start: datetime | None = None,
    ) -> TraceCarbonIntensityProvider:
        """Load a balancing-authority workbook from the U.S. EIA Hourly Electric Grid Monitor, as it is.

        These are the `.xlsx` files under Download Data, Balancing Authority/Regional
        files (for example `ERCO.xlsx`). The `Published Hourly Data` sheet has `UTC time`
        (the end of each hour) and the CO2 emissions intensity of consumed or generated
        electricity in lb/kWh (direct emissions, from July 2018). Hours without a CO2
        estimate are skipped. Reading a full-history workbook takes about 10 seconds with
        `python-calamine` installed, or about 35 seconds with `openpyxl`.

        Args:
            path: Path to the workbook.
            electricity: `"consumed"` (accounts for imports and exports) or `"generated"`.
            replay_start: Passed to the constructor.
        """
        column = f"CO2 Emissions Intensity for {electricity.capitalize()} Electricity"
        engine = "calamine" if importlib.util.find_spec("python_calamine") else None
        try:
            df = pd.read_excel(path, sheet_name="Published Hourly Data", engine=engine)
        except ImportError as e:
            raise ImportError("Reading EIA-930 workbooks needs `pip install python-calamine` (or openpyxl).") from e
        return cls._from_series(
            pd.to_datetime(df["UTC time"]).dt.tz_localize("UTC"),
            df[column],
            scale=453.59237,  # lb/kWh to g/kWh
            time_label="end",
            period=timedelta(hours=1),
            replay_start=replay_start,
        )

    @classmethod
    def from_gb_carbon_intensity_json(
        cls,
        path: str | Path,
        value: Literal["actual", "forecast"] = "actual",
        replay_start: datetime | None = None,
    ) -> TraceCarbonIntensityProvider:
        """Load a saved response from the GB Carbon Intensity API (api.carbonintensity.org.uk), as it is.

        Takes the national `/intensity/{from}/{to}` response: 30-minute periods with
        `from`, `to` and `intensity.actual` / `intensity.forecast` in gCO2/kWh. Periods
        without the chosen value are skipped.

        Args:
            path: Path to the saved JSON response.
            value: `"actual"` or `"forecast"`.
            replay_start: Passed to the constructor.
        """
        with open(path) as fh:
            data = json.load(fh)["data"]
        times = pd.to_datetime([d["from"] for d in data], utc=True)
        values = [d["intensity"].get(value) for d in data]
        return cls._from_series(
            pd.Series(times),
            pd.Series(values, dtype="float64"),
            period=timedelta(minutes=30),
            replay_start=replay_start,
        )

    def carbon_intensity_at(self, when: datetime) -> float:
        """Return the carbon intensity of the trace period that contains `when` (a trace time, not wall-clock)."""
        if when.tzinfo is None:
            raise ValueError("`when` must be timezone-aware.")
        start = _period_floor(when, self.period)
        try:
            return self.trace[start]
        except KeyError as e:
            first, last = next(iter(self.trace)), next(reversed(self.trace))
            raise ZeusCarbonIntensityNotFoundError(
                f"No carbon intensity for the period starting {start.isoformat()}. "
                f"The trace covers {first.isoformat()} to {last.isoformat()} (period starts, UTC), "
                "and may have gaps."
            ) from e

    def get_current_carbon_intensity(self) -> float:
        """Return the carbon intensity of the trace period mapped to the current wall-clock period."""
        return self.carbon_intensity_at(datetime.now(timezone.utc) + self.offset)

    def get_recent_carbon_intensity(self) -> dict[datetime, float]:
        """Return every period of the trace, keyed by the wall-clock period it is mapped to."""
        return {start - self.offset: value for start, value in self.trace.items()}

    @property
    def update_period(self) -> timedelta:
        """Length of each trace period."""
        return self.period

    @property
    def history_length(self) -> int:
        """Number of periods in the trace, since `get_recent_carbon_intensity` returns all of them."""
        return len(self.trace)


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _period_floor(when: datetime, period: timedelta) -> datetime:
    """Return the start of the period of length `period` (on a UTC grid from the Unix epoch) that contains `when`."""
    return _EPOCH + ((when.astimezone(timezone.utc) - _EPOCH) // period) * period


@dataclass
class CarbonEmissionMeasurement:
    """Measurement result of one window.

    Attributes:
        time: Time elapsed (in seconds) during the measurement window.
        gpu_energy: Maps GPU indices to the energy consumed (in Joules) during the
            measurement window. GPU indices are from the DL framework's perspective
            after applying `CUDA_VISIBLE_DEVICES`.
        gpu_carbon_emission: Maps GPU indices to the carbon emission produced (in gCO2eq) during the
            measurement window. GPU indices are from the DL framework's perspective
            after applying `CUDA_VISIBLE_DEVICES`.
        cpu_energy: Maps CPU indices to the energy consumed (in Joules) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d). This can
            be 'None' if CPU measurement is not available.
        cpu_carbon_emission: Maps CPU indices to the carbon emission produced (in gCO2eq) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d). This can
            be 'None' if CPU measurement is not available.
        dram_energy: Maps CPU indices to the energy consumed (in Joules) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d) and DRAM
            measurements are taken from sub-packages within each powerzone. This can be 'None' if
            CPU measurement is not available or DRAM measurement is not available.
        dram_carbon_emission: Maps CPU indices to the carbon emission produced (in gCO2eq) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d). This can be 'None' if
            CPU measurement is not available or DRAM measurement is not available.
    """

    time: float
    gpu_energy: dict[int, float]
    gpu_carbon_emission: dict[int, float]
    cpu_energy: dict[int, float] | None = None
    cpu_carbon_emission: dict[int, float] | None = None
    dram_energy: dict[int, float] | None = None
    dram_carbon_emission: dict[int, float] | None = None


class Op(Enum):
    """Enum used to communicate between CarbonEmissionMonitor and _polling_process."""

    BEGIN = 0
    END = 1
    NEXTITER = 2


class CarbonEmissionMonitor:
    """Measure the carbon emission, GPU energy, and time consumption of a block of code.

    Works for multi-GPU and heterogeneous GPU types. Aware of `CUDA_VISIBLE_DEVICES`.
    For instance, if `CUDA_VISIBLE_DEVICES=2,3`, GPU index `1` passed into `gpu_indices`
    will be interpreted as CUDA device `3`.

    You can mark the beginning and end of a measurement window, during which the carbon
    emission, GPU energy, and time consumed will be recorded. Multiple concurrent
    measurement windows are supported.

    !!! Note
        `carbon_intensity_provider` must have `estimate` turned on because during some hours,
        ElectricityMaps does not have carbon intensity values available and has to rely on
        estimation.

    !!! Warning
        This monitor uses multiprocessing with the spawn start method to measure energy and carbon emissions
        in a background process. Spawned processes re-import your main module, so keep
        heavy setup under `if __name__ == "__main__":` or inside functions.
        See also the "Safe importing of main module" section in the [Python documentation](
        https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods).
    """

    def __init__(
        self,
        carbon_intensity_provider: CarbonIntensityProvider,
        gpu_indices: list[int] | None = None,
        cpu_indices: list[int] | None = None,
        sync_execution_with: Literal["torch", "jax", "cupy"] = "torch",
    ) -> None:
        """Initializes Carbon Emission Monitor.

        Args:
            carbon_intensity_provider: provider for which carbon intensity values will be fetched from
            gpu_indices: Indices of all the CUDA devices to monitor. Time/Energy measurements
                will begin and end at the same time for these GPUs (i.e., synchronized).
                If None, all the GPUs available will be used. `CUDA_VISIBLE_DEVICES`
                is respected if set, e.g., GPU index `1` passed into `gpu_indices` when
                `CUDA_VISIBLE_DEVICES=2,3` will be interpreted as CUDA device `3`.
                `CUDA_VISIBLE_DEVICES`s formatted with comma-separated indices are supported.
            cpu_indices: Indices of the CPU packages to monitor. If None, all CPU packages will
                be used.
            sync_execution_with: Deep learning framework to use to synchronize CPU/GPU computations.
                Defaults to `"torch"`, in which case `torch.cuda.synchronize` will be used.
                See [`sync_execution`][zeus.utils.framework.sync_execution] for more details.
        """
        # Warn if instantiated as a global variable in a subprocess.
        warn_if_global_in_subprocess(self)

        self.zeus_monitor = ZeusMonitor(
            gpu_indices=gpu_indices,
            cpu_indices=cpu_indices,
            sync_execution_with=sync_execution_with,
        )
        self.carbon_intensity_provider = carbon_intensity_provider
        self.current_keys = set()

        # set up process and shared queues
        self.context = mp.get_context("spawn")
        self.command_q = self.context.Queue()
        self.finished_q = self.context.Queue()

    def begin_window(self, key: str, sync_execution: bool = True) -> None:
        """Begin a new measurement window.

        Args:
            key: Unique name of the measurement window.
            sync_execution: Whether to wait for asynchronously dispatched computations
                to finish before starting the measurement window. For instance, PyTorch
                and JAX will run GPU computations asynchronously, and waiting them to
                finish is necessary to ensure that the measurement window captures all
                and only the computations dispatched within the window.
        """
        # check if key is already used
        if key in self.current_keys:
            raise ValueError(f"Measurement window '{key}' already exists")
        self.current_keys.add(key)

        # start window
        self.zeus_monitor.begin_window(key, sync_execution=sync_execution)

        # if there were previously no active windows, start polling process
        if len(self.current_keys) == 1:
            self.polling_process = self.context.Process(
                target=_polling_process,
                args=(
                    self.command_q,
                    self.finished_q,
                    self.zeus_monitor.gpu_indices,
                    self.zeus_monitor.cpu_indices,
                    self.carbon_intensity_provider,
                ),
            )
            self.polling_process.start()

        # start subwindows
        self.command_q.put((Op.BEGIN, key))

    def end_window(self, key: str, sync_execution: bool = True) -> CarbonEmissionMeasurement:
        """End a measurement window and return the time, energy consumption, and carbon emission.

        Args:
            key: Name of an active measurement window.
            sync_execution: Whether to wait for asynchronously dispatched computations
                to finish before starting the measurement window. For instance, PyTorch
                and JAX will run GPU computations asynchronously, and waiting them to
                finish is necessary to ensure that the measurement window captures all
                and only the computations dispatched within the window.
        """
        # check if begin_window has been called with key before
        if key not in self.current_keys:
            raise ValueError(f"Measurement window '{key}' does not exist")

        # end window
        self.command_q.put((Op.END, key))
        (
            gpu_carbon_emissions,
            cpu_carbon_emissions,
            dram_carbon_emissions,
        ) = self.finished_q.get()
        self.current_keys.remove(key)

        overall_measurement = self.zeus_monitor.end_window(key, sync_execution=sync_execution)

        measurement = CarbonEmissionMeasurement(
            time=overall_measurement.time,
            gpu_energy=overall_measurement.gpu_energy,
            cpu_energy=overall_measurement.cpu_energy,
            dram_energy=overall_measurement.dram_energy,
            gpu_carbon_emission=gpu_carbon_emissions,
            cpu_carbon_emission=cpu_carbon_emissions or None,
            dram_carbon_emission=dram_carbon_emissions or None,
        )

        return measurement


def _polling_process(
    command_q: mp.Queue,
    finished_q: mp.Queue,
    gpu_indices: list[int],
    cpu_indices: list[int],
    carbon_intensity_provider: CarbonIntensityProvider,
):
    last_index = 0
    index = 0
    zeus_monitor = ZeusMonitor(gpu_indices=gpu_indices, cpu_indices=cpu_indices)
    gpu_carbon_emissions = defaultdict(
        lambda: defaultdict(float)
    )  # {window_key -> {gpu index -> cumulative carbon emission}}
    cpu_carbon_emissions = defaultdict(
        lambda: defaultdict(float)
    )  # {window_key -> {cpu index -> cumulative carbon emission}}
    dram_carbon_emissions = defaultdict(
        lambda: defaultdict(float)
    )  # {window_key -> {dram cpu index -> cumulative carbon emission}}
    energy_measurements = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )  # {window_key -> {hour -> {gpu/cpu index -> energy}}}
    keys = set()

    # record energy measurement
    def _update_energy_measurements(key: str, datetime: datetime):
        measurement = zeus_monitor.end_window(key, sync_execution=False)

        for gpu_index, energy_measurement in measurement.gpu_energy.items():
            energy_measurements[key][datetime][f"gpu_{gpu_index}"] = energy_measurement

        if measurement.cpu_energy:
            for cpu_index, energy_measurement in measurement.cpu_energy.items():
                energy_measurements[key][datetime][f"cpu_{cpu_index}"] = energy_measurement

        if measurement.dram_energy:
            for dram_index, energy_measurement in measurement.dram_energy.items():
                energy_measurements[key][datetime][f"dram_{dram_index}"] = energy_measurement

    # update cumulative carbon emissions
    def _update_carbon_emissions(key: str):
        carbon_intensities = carbon_intensity_provider.get_recent_carbon_intensity()

        for dt, measurement_map in energy_measurements[key].items():
            for index, energy in measurement_map.items():
                hardware_type, num_index = index.split("_")

                if hardware_type == "gpu":
                    gpu_carbon_emissions[key][int(num_index)] += energy / 3.6e6 * carbon_intensities[dt]
                elif hardware_type == "cpu":
                    cpu_carbon_emissions[key][int(num_index)] += energy / 3.6e6 * carbon_intensities[dt]
                elif hardware_type == "dram":
                    dram_carbon_emissions[key][int(num_index)] += energy / 3.6e6 * carbon_intensities[dt]

        del energy_measurements[key]

    while True:
        # calculate current time
        now = datetime.now(timezone.utc)
        # Periods of `update_period` on a UTC grid (whole hours for hourly providers).
        hour_floor = _period_floor(now, carbon_intensity_provider.update_period)
        hour_ceil = hour_floor + carbon_intensity_provider.update_period

        # start windows
        for key in keys:
            zeus_monitor.begin_window(key, sync_execution=False)

        try:
            # continuously fetch from command q until hit hour
            while now < hour_ceil:
                seconds_until_hour = (hour_ceil - now).total_seconds()
                op, key = command_q.get(timeout=seconds_until_hour)

                if op == Op.BEGIN:
                    zeus_monitor.begin_window(key, sync_execution=False)
                    keys.add(key)
                elif op == Op.END:
                    # update if has not been updated in a while
                    _update_energy_measurements(key, hour_floor)
                    _update_carbon_emissions(key)
                    finished_q.put(
                        (
                            dict(gpu_carbon_emissions.pop(key)),
                            dict(cpu_carbon_emissions.pop(key)),
                            dict(dram_carbon_emissions.pop(key)),
                        )
                    )
                    keys.remove(key)

                    if len(keys) == 0:
                        return
                elif op == Op.NEXTITER:
                    # for testing purposes only, force monitor to move onto next hour
                    # this op will never be sent into command q outside of testing environments
                    break

                now = datetime.now(timezone.utc)
        except queue.Empty:
            # if nothing from finished_q.get(), continue
            pass

        # record energy values
        index += 1
        for key in keys:
            _update_energy_measurements(key, hour_floor)

        # check if we need to fetch carbon intensity measurements
        # only need to fetch if oldest energy measurement's datetime is about to be outside the time range covered by fetched carbon intensity values
        if index - last_index == carbon_intensity_provider.history_length - 1:
            for key in keys:
                _update_carbon_emissions(key)
            last_index = index
