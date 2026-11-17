"""Carbon intensity providers used for carbon-aware optimizers.

[`ElectrictyMapsClient`][zeus.monitor.carbon.ElectrictyMapsClient] fetches live hourly carbon
intensity from the Electricity Maps API.
[`TraceCarbonIntensityProvider`][zeus.monitor.carbon.TraceCarbonIntensityProvider] reads carbon
intensity from a downloaded file, which is useful for reproducing a
[`CarbonEmissionMonitor`][zeus.monitor.carbon.CarbonEmissionMonitor] result or replaying a past
day. Supported formats:

- Electricity Maps CSV exports (hourly):
  [`from_electricitymaps_csv`][zeus.monitor.carbon.TraceCarbonIntensityProvider.from_electricitymaps_csv].
- U.S. EIA-930 Hourly Electric Grid Monitor balancing-authority workbooks (`.xlsx`, hourly):
  [`from_eia930_xlsx`][zeus.monitor.carbon.TraceCarbonIntensityProvider.from_eia930_xlsx].
- GB Carbon Intensity API responses saved as JSON (30-minute intervals, past data or
  48-hour forecasts):
  [`from_gb_carbon_intensity_json`][zeus.monitor.carbon.TraceCarbonIntensityProvider.from_gb_carbon_intensity_json].
- Other CSV files with one timestamp column and one intensity column:
  [`from_csv`][zeus.monitor.carbon.TraceCarbonIntensityProvider.from_csv].
"""

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
    """Carbon intensity provider backed by a downloaded trace.

    The trace is a time series of carbon intensity in gCO2eq/kWh. It is grouped into
    periods of length `period` (for example one hour) on a UTC grid. A lookup returns the
    value of the period that contains the requested time. Values are not interpolated or
    filled in, and `carbon_intensity_at` and `get_current_carbon_intensity` raise
    `ZeusCarbonIntensityNotFoundError` for periods without data.

    To compute a run's emissions from past data, pass `replay_start`. Otherwise the trace
    must cover the run itself, for example a GB Carbon Intensity 48-hour forecast
    (`/intensity/{from}/fw48h`) saved just before the run and loaded with `value="forecast"`.

    The `from_*` class methods load common file formats.

    !!! Note
        Do not mix direct (combustion) and lifecycle emission factors when you compare
        results.
    """

    def __init__(
        self,
        intensities: Mapping[datetime, float],
        period: timedelta,
        data_spacing: timedelta,
        replay_start: datetime | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            intensities: Maps the start of each data point (timezone-aware) to the carbon
                intensity in gCO2eq/kWh. Values must be finite and non-negative. Every timestamp
                must be on a UTC grid of `data_spacing` (for example, the top of each hour for
                hourly data). Gaps are allowed.
            period: Length of the periods carbon is accounted in. Must be a whole multiple
                of the data's spacing, so the data can't be coarser than `period`. Each
                period gets the mean of its data points, and periods missing any point are
                left out.
            data_spacing: Time between data points, for example one hour for hourly data.
            replay_start: A timezone-aware time inside the trace. If given, the wall-clock
                period in which the provider is created is mapped to the trace period that
                contains this time, and later periods follow one to one.
        """
        if period <= timedelta(0):
            raise ValueError("`period` must be positive.")
        if data_spacing <= timedelta(0):
            raise ValueError("`data_spacing` must be positive.")
        if data_spacing > period:
            raise ValueError(
                f"The data is coarser than `period`: `data_spacing` is {data_spacing}, but `period` is {period}."
            )
        if period % data_spacing:
            raise ValueError(f"`period` ({period}) must be a whole multiple of `data_spacing` ({data_spacing}).")
        if not intensities:
            raise ValueError("The carbon intensity trace is empty.")
        points: dict[datetime, float] = {}
        for start, value in intensities.items():
            if start.tzinfo is None:
                raise ValueError(f"Trace time {start} has no timezone. Use timezone-aware times.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Carbon intensity {value} at {start} is not a finite non-negative number.")
            points[start.astimezone(timezone.utc)] = float(value)

        starts = sorted(points)
        for start in starts:
            if _period_floor(start, data_spacing) != start:
                raise ValueError(
                    f"Trace time {start} is not on the {data_spacing} grid (for example, the top of each hour "
                    "for hourly data)."
                )

        per_period = period // data_spacing
        grouped: dict[datetime, list[float]] = defaultdict(list)
        for start in starts:
            grouped[_period_floor(start, period)].append(points[start])
        self.period = period
        self.trace = {p: sum(v) / len(v) for p, v in sorted(grouped.items()) if len(v) == per_period}
        if not self.trace:
            raise ValueError(f"No {period} period is fully covered by the data.")
        if len(self.trace) < len(grouped):
            logger.warning(
                "Left out %d of %d periods of length %s that the data covers only partly.",
                len(grouped) - len(self.trace),
                len(grouped),
                period,
            )

        self.replay_start = replay_start
        self.offset = timedelta(0)
        if replay_start is None:
            now = _period_floor(datetime.now(timezone.utc), period)
            if now not in self.trace:
                logger.warning(
                    "The trace has no value for the current period (%s). Pass `replay_start` to replay past data.",
                    now.isoformat(),
                )
        else:
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
        period: timedelta,
        data_spacing: timedelta,
        scale: float = 1.0,
        time_label: Literal["start", "end"] = "start",
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
            # Each timestamp ends one data interval, so move it back by `data_spacing`.
            times = times - data_spacing
        trace = {t.to_pydatetime(): float(v) for t, v in zip(times, values)}
        return cls(trace, period=period, data_spacing=data_spacing, replay_start=replay_start)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        time_column: str,
        intensity_column: str,
        period: timedelta,
        data_spacing: timedelta,
        time_zone: str | None = None,
        time_label: Literal["start", "end"] = "start",
        scale: float = 1.0,
        replay_start: datetime | None = None,
    ) -> TraceCarbonIntensityProvider:
        """Build the provider from a CSV file with one timestamp column and one intensity column.

        The other columns are ignored.

        A file with UTC offsets in its timestamps and values in gCO2eq/kWh, stamped at the
        start of each hour:

        ```
        timestamp,carbon_intensity
        2024-07-19T00:00:00+00:00,383.4
        2024-07-19T01:00:00+00:00,379.6
        ```

        ```python
        TraceCarbonIntensityProvider.from_csv(
            "trace.csv",
            "timestamp",
            "carbon_intensity",
            period=timedelta(hours=1),
            data_spacing=timedelta(hours=1),
        )
        ```

        A file with UTC times without offsets, every 15 minutes, accounted in hourly periods
        (each hour gets the mean of its four values):

        ```
        datetime_utc,gco2_per_kwh
        2024-07-19 00:00,383.4
        2024-07-19 00:15,381.0
        ...
        ```

        ```python
        TraceCarbonIntensityProvider.from_csv(
            "trace.csv",
            "datetime_utc",
            "gco2_per_kwh",
            period=timedelta(hours=1),
            data_spacing=timedelta(minutes=15),
            time_zone="UTC",
        )
        ```

        A file with local times without offsets and values in lb/kWh, stamped at the end of
        each hour:

        ```
        Local time,CO2 intensity (lb/kWh)
        2024-07-19 01:00:00,0.84
        2024-07-19 02:00:00,0.83
        ```

        ```python
        TraceCarbonIntensityProvider.from_csv(
            "trace.csv",
            "Local time",
            "CO2 intensity (lb/kWh)",
            period=timedelta(hours=1),
            data_spacing=timedelta(hours=1),
            time_zone="America/Chicago",
            time_label="end",
            scale=453.59237,  # lb/kWh to g/kWh
        )
        ```

        Args:
            path: Path to the CSV file.
            time_column: Name of the timestamp column.
            intensity_column: Name of the carbon intensity column.
            period: Length of the periods carbon is accounted in. Must be a whole multiple
                of the data's spacing, so the data can't be coarser than `period`. Each
                period gets the mean of its data points, and periods missing any point are
                left out.
            data_spacing: Time between rows, for example one hour for hourly data.
            time_zone: IANA time zone (for example `"UTC"` or `"America/Chicago"`) for
                timestamps without an offset. Must be omitted if the timestamps have an
                offset (such as `Z`, `+00:00` or ` UTC`). Prefer UTC timestamps, since local
                times are ambiguous when daylight saving time ends.
            time_label: Whether each timestamp marks the `"start"` or the `"end"` of its interval.
            scale: Factor that converts the column to gCO2eq/kWh. For example, 453.59237
                for lb/kWh, or 1000 for t/MWh.
            replay_start: A timezone-aware time inside the trace. If given, the wall-clock
                period in which the provider is created is mapped to the trace period that
                contains this time, and later periods follow one to one.
        """
        df = pd.read_csv(path)
        for column in (time_column, intensity_column):
            if column not in df.columns:
                raise ValueError(f"Column '{column}' is not in {path}.")
        text = df[time_column].astype(str).str.strip()
        # An offset after the time of day: Z, UTC, +00, +0530 or +05:30.
        has_offset = text.str.contains(r"\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:Z|UTC|[+-]\d{2}(?::?\d{2})?)$")
        if has_offset.all():
            if time_zone is not None:
                raise ValueError(f"Column '{time_column}' already has UTC offsets. Do not pass `time_zone`.")
            times = _parse_times(text, utc=True)
        elif not has_offset.any():
            if time_zone is None:
                raise ValueError(f"Column '{time_column}' has no UTC offsets. Pass `time_zone`.")
            times = _parse_times(text).dt.tz_localize(time_zone)
        else:
            raise ValueError(f"Column '{time_column}' mixes times with and without UTC offsets.")
        return cls._from_series(times, df[intensity_column], period, data_spacing, scale, time_label, replay_start)

    @classmethod
    def from_electricitymaps_csv(
        cls,
        path: str | Path,
        period: timedelta,
        data_spacing: timedelta,
        emission_factor_type: Literal["direct", "lifecycle"] = "direct",
        replay_start: datetime | None = None,
    ) -> TraceCarbonIntensityProvider:
        """Load a CSV file exported from Electricity Maps.

        Reads `Datetime (UTC)` and the direct or lifecycle carbon intensity column. Both
        the older headers (`Carbon Intensity gCO₂eq/kWh (LCA)`) and the newer headers
        (`Carbon intensity gCO₂eq/kWh (Life cycle)`) work. Use the hourly exports, with
        `data_spacing=timedelta(hours=1)`.

        Args:
            path: Path to the CSV file.
            period: Length of the periods carbon is accounted in. Must be a whole multiple
                of the data's spacing, so the data can't be coarser than `period`. Each
                period gets the mean of its data points, and periods missing any point are
                left out.
            data_spacing: Time between rows, for example one hour for the hourly exports.
            emission_factor_type: `"direct"` or `"lifecycle"`. The default is `"direct"`, as in
                `ElectrictyMapsClient`.
            replay_start: A timezone-aware time inside the trace. If given, the wall-clock
                period in which the provider is created is mapped to the trace period that
                contains this time, and later periods follow one to one.
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
        times = pd.to_datetime(df["Datetime (UTC)"], utc=True)
        return cls._from_series(times, df[column], period, data_spacing, replay_start=replay_start)

    @classmethod
    def from_eia930_xlsx(
        cls,
        path: str | Path,
        period: timedelta,
        electricity: Literal["consumed", "generated"] = "consumed",
        replay_start: datetime | None = None,
    ) -> TraceCarbonIntensityProvider:
        """Load a balancing-authority workbook from the U.S. EIA Hourly Electric Grid Monitor.

        These are the `.xlsx` files under Download Data, Balancing Authority/Regional
        files (for example `ERCO.xlsx`). The `Published Hourly Data` sheet has `UTC time`
        (the end of each hour) and the CO2 emissions intensity of consumed or generated
        electricity in lb/kWh (direct emissions, from July 2018). Hours without a CO2
        estimate are skipped. Reading `.xlsx` needs `python-calamine` (used if installed)
        or `openpyxl`.

        Args:
            path: Path to the workbook.
            period: Length of the periods carbon is accounted in. Must be a whole multiple
                of the data's spacing, so the data can't be coarser than `period`. Each
                period gets the mean of its data points, and periods missing any point are
                left out.
            electricity: `"consumed"` (accounts for imports and exports) or `"generated"`.
            replay_start: A timezone-aware time inside the trace. If given, the wall-clock
                period in which the provider is created is mapped to the trace period that
                contains this time, and later periods follow one to one.
        """
        column = f"CO2 Emissions Intensity for {electricity.capitalize()} Electricity"
        engine = "calamine" if importlib.util.find_spec("python_calamine") else None
        try:
            df = pd.read_excel(path, sheet_name="Published Hourly Data", engine=engine)
        except ImportError as e:
            raise ImportError("Reading EIA-930 workbooks needs `pip install python-calamine` (or openpyxl).") from e
        if "UTC time" not in df.columns or column not in df.columns:
            raise ValueError(
                f"{path} does not look like an EIA-930 balancing-authority workbook: expected 'UTC time' and "
                f"'{column}' in the 'Published Hourly Data' sheet."
            )
        return cls._from_series(
            pd.to_datetime(df["UTC time"], utc=True),
            df[column],
            period,
            scale=453.59237,  # lb/kWh to g/kWh
            time_label="end",
            replay_start=replay_start,
            data_spacing=timedelta(hours=1),
        )

    @classmethod
    def from_gb_carbon_intensity_json(
        cls,
        path: str | Path,
        period: timedelta,
        value: Literal["actual", "forecast"] = "actual",
        replay_start: datetime | None = None,
    ) -> TraceCarbonIntensityProvider:
        """Load a saved response from the GB Carbon Intensity API (api.carbonintensity.org.uk).

        Takes national `/intensity` responses, such as `/intensity/{from}/{to}` for past
        data or `/intensity/{from}/fw48h` for a 48-hour forecast: 30-minute intervals with
        `from`, `to` and `intensity.actual` / `intensity.forecast` in gCO2/kWh. Intervals
        without the chosen value are skipped.

        Args:
            path: Path to the saved JSON response.
            period: Length of the periods carbon is accounted in. Must be a whole multiple
                of the data's spacing, so the data can't be coarser than `period`. Each
                period gets the mean of its data points, and periods missing any point are
                left out.
            value: `"actual"` or `"forecast"`.
            replay_start: A timezone-aware time inside the trace. If given, the wall-clock
                period in which the provider is created is mapped to the trace period that
                contains this time, and later periods follow one to one.
        """
        with open(path) as fh:
            data = json.load(fh)["data"]
        times = pd.to_datetime([d["from"] for d in data], utc=True)
        values = [d["intensity"].get(value) for d in data]
        return cls._from_series(
            pd.Series(times),
            pd.Series(values, dtype="float64"),
            period,
            replay_start=replay_start,
            data_spacing=timedelta(minutes=30),
        )

    def carbon_intensity_at(self, when: datetime) -> float:
        """Return the carbon intensity of the trace period that contains `when`, in trace time."""
        if when.tzinfo is None:
            raise ValueError("`when` must be timezone-aware.")
        start = _period_floor(when, self.period)
        try:
            return self.trace[start]
        except KeyError as e:
            first, last = next(iter(self.trace)), next(reversed(self.trace))
            message = (
                f"No carbon intensity for the period starting {start.isoformat()}. "
                f"The trace covers {first.isoformat()} to {last.isoformat()} (UTC) and may have gaps."
            )
            if self.replay_start is None:
                message += " Pass `replay_start` to replay past data."
            raise ZeusCarbonIntensityNotFoundError(message) from e

    def get_current_carbon_intensity(self) -> float:
        """Return the carbon intensity of the trace period mapped to the current wall-clock period."""
        return self.carbon_intensity_at(datetime.now(timezone.utc) + self.offset)

    def get_recent_carbon_intensity(self) -> dict[datetime, float]:
        """Return every period of the trace, keyed by the wall-clock period it is mapped to."""
        return {start - self.offset: value for start, value in self.trace.items()}

    @property
    def update_period(self) -> timedelta:
        """Length of each period."""
        return self.period

    @property
    def history_length(self) -> int:
        """Number of periods in the trace."""
        return len(self.trace)


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _period_floor(when: datetime, period: timedelta) -> datetime:
    """Return the start of the period of length `period` (on a UTC grid from the Unix epoch) that contains `when`."""
    return _EPOCH + ((when.astimezone(timezone.utc) - _EPOCH) // period) * period


def _parse_times(text: pd.Series, utc: bool = False) -> pd.Series:
    """Parse timestamps, also when ISO 8601 rows differ in precision.

    pandas guesses one format from the first row, so `00:00:00Z` followed by `01:00:00.000Z` fails.
    """
    try:
        return pd.to_datetime(text, utc=utc)
    except ValueError:
        return pd.to_datetime(text, utc=utc, format="ISO8601")


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
