"""Carbon intensity providers used for carbon-aware optimizers."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum
import queue
import requests
import multiprocessing as mp

from typing import Literal
from datetime import datetime, timezone, timedelta
from dateutil import parser
from collections import defaultdict

from zeus.exception import ZeusBaseError
from zeus.monitor import ZeusMonitor
from zeus.utils.logging import get_logger

logger = get_logger(__name__)


def get_ip_lat_long() -> tuple[float, float]:
    """Retrieve the latitude and longitude of the current IP position."""
    try:
        ip_url = "http://ipinfo.io/json"
        resp = requests.get(ip_url)
        loc = resp.json()["loc"]
        lat, long = map(float, loc.split(","))
        logger.info("Retrieved latitude and longitude: %s, %s", lat, long)
        return lat, long
    except requests.exceptions.RequestException as e:
        logger.exception(
            "Failed to retrieve current latitude and longitude of IP: %s", e
        )
        raise


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
            raise ZeusCarbonIntensityHTTPError(
                f"Failed to retrieve current carbon intensity measurement: {e}"
            ) from e

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
            raise ZeusCarbonIntensityHTTPError(
                f"Failed to retrieve recent carbon intensity measurement: {e}"
            ) from e

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
    """

    def __init__(
        self,
        carbon_intensity_provider: CarbonIntensityProvider,
        gpu_indices: list[int] | None = None,
        cpu_indices: list[int] | None = None,
        sync_execution_with: Literal["torch", "jax"] = "torch",
    ) -> None:
        """Iniitializes Carbon Emission Monitor.

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

    def end_window(
        self, key: str, sync_execution: bool = True
    ) -> CarbonEmissionMeasurement:
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

        overall_measurement = self.zeus_monitor.end_window(
            key, sync_execution=sync_execution
        )

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
                energy_measurements[key][datetime][
                    f"cpu_{cpu_index}"
                ] = energy_measurement

        if measurement.dram_energy:
            for dram_index, energy_measurement in measurement.dram_energy.items():
                energy_measurements[key][datetime][
                    f"dram_{dram_index}"
                ] = energy_measurement

    # update cumulative carbon emissions
    def _update_carbon_emissions(key: str):
        carbon_intensities = carbon_intensity_provider.get_recent_carbon_intensity()

        for dt, measurement_map in energy_measurements[key].items():
            for index, energy in measurement_map.items():
                hardware_type, num_index = index.split("_")

                if hardware_type == "gpu":
                    gpu_carbon_emissions[key][int(num_index)] += (
                        energy / 3.6e6 * carbon_intensities[dt]
                    )
                elif hardware_type == "cpu":
                    cpu_carbon_emissions[key][int(num_index)] += (
                        energy / 3.6e6 * carbon_intensities[dt]
                    )
                elif hardware_type == "dram":
                    dram_carbon_emissions[key][int(num_index)] += (
                        energy / 3.6e6 * carbon_intensities[dt]
                    )

        del energy_measurements[key]

    while True:
        # calculate current time
        now = datetime.now(timezone.utc)
        hour_floor = now.replace(minute=0, second=0, microsecond=0)
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
