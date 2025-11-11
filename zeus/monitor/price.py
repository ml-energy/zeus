"""Electricity price providers used for price-aware optimizers."""

from __future__ import annotations

import abc
import queue
import requests
import json
import logging
import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum
from typing import Literal
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from zeus.exception import ZeusBaseError
from zeus.monitor import ZeusMonitor

logger = logging.getLogger(__name__)


def get_time_info() -> tuple[str, str, int]:
    """Retrieve the month, day_type (weekend or weekday), and hour."""
    now = datetime.now()

    month = now.strftime("%B")

    day_type = "Weekend" if now.weekday() >= 5 else "Weekday"

    hour = now.hour

    return month, day_type, hour


class ZeusElectricityPriceHTTPError(ZeusBaseError):
    """Exception when HTTP request to electricity price provider fails."""

    def __init__(self, message: str) -> None:
        """Initialize HTTP request exception."""
        super().__init__(message)


class ZeusElectricityPriceNotFoundError(ZeusBaseError):
    """Exception when electricity price measurement could not be retrieved."""

    def __init__(self, message: str) -> None:
        """Initialize price not found exception."""
        super().__init__(message)


class ElectricityPriceProvider(abc.ABC):
    """Abstract class for implementing ways to fetch electricity price."""

    @abc.abstractmethod
    def get_current_electricity_prices(self) -> dict[str, list]:
        """Abstract method for fetching the current electricity price of the set location of the class."""
        pass


class OpenEIClient(ElectricityPriceProvider):
    """Electricity Price Provider with OpenEI API.

    Reference:

    1. [OpenEI](https://openei.org/wiki/Main_Page)
    2. [OpenEI Utility Rates API](https://apps.openei.org/services/doc/rest/util_rates/?version=7)
    """

    def __init__(
        self,
        location: tuple[float, float],
        label: str,
        sector: Literal["Residential", "Commercial", "Industrial", "Lighting"] = "Residential",
        radius: int = 0,
    ) -> None:
        """Initializes OpenEI Utility Rates Provider.

        Args:
            location: tuple of latitude and longitude (latitude, longitude)
            label: unique identifier of a particular variant of a utility company's rate
            sector: depends on which sector of electricity is relevant to you
            radius: search radius for utility rates from the location
        """
        self.lat, self.long = location
        self.label = label
        self.sector = sector
        self.radius = radius

    def search_json(self, data, key_name, target_value, return_value):
        """Recursively search for a key in a nested JSON and return the return_value field if found."""
        results = []

        if isinstance(data, dict):
            for key, val in data.items():
                # Check if the current dictionary contains the matching key-value pair
                if key == key_name and val == target_value:
                    # If "energyratestructure" exists at the same level, add it to results
                    if return_value in data:
                        results.append(data[return_value])
                    else:
                        results.append(None)

                # Recursively search deeper in nested dictionaries
                results.extend(self.search_json(val, key_name, target_value, return_value))

        elif isinstance(data, list):
            for item in data:
                results.extend(self.search_json(item, key_name, target_value, return_value))

        return results

    def get_current_electricity_prices(self) -> dict[str, list]:
        """Fetches current carbon intensity of the location of the class."""
        try:
            url = (
                "https://api.openei.org/utility_rates?version=latest&format=json"
                + f"&api_key=tJASWWgPhBRpiZCwfhtKV2A3gyNxbDfvQvdI5Wa7&lat={self.lat}"
                + f"&lon={self.long}&radius={self.radius}"
                + f"&detail=full&sector={self.sector}"
            )
            resp = requests.get(url)
            data = resp.json()

        except requests.exceptions.RequestException as e:
            raise ZeusElectricityPriceHTTPError(f"Failed to retrieve current electricity price measurement: {e}") from e

        try:
            if "label" not in json.dumps(data):
                raise ZeusElectricityPriceNotFoundError(f"No rates found for lat, lon: [{self.lat}, {self.long}].")

            energy_rate_structure = self.search_json(data, "label", self.label, "energyratestructure")
            energy_weekday_schedule = self.search_json(data, "label", self.label, "energyweekdayschedule")
            energy_weekend_schedule = self.search_json(data, "label", self.label, "energyweekendschedule")

            if not energy_rate_structure or not energy_weekday_schedule or not energy_weekend_schedule:
                raise ZeusElectricityPriceNotFoundError(f"No rates found for the label: {self.label}.")

            rate_data = {
                "energy_rate_structure": energy_rate_structure[0],
                "energy_weekday_schedule": energy_weekday_schedule[0],
                "energy_weekend_schedule": energy_weekend_schedule[0],
            }
            return rate_data

        except (KeyError, ValueError) as e:
            logger.error("Error occurred while processing electricity price data: %s", e)
            raise ZeusElectricityPriceNotFoundError("Failed to process electricity price data.") from e


@dataclass
class EnergyCostMeasurement:
    """Measurement result of one window.

    Attributes:
        time: Time elapsed (in seconds) during the measurement window.
        gpu_energy: Maps GPU indices to the energy consumed (in Joules) during the
            measurement window. GPU indices are from the DL framework's perspective
            after applying `CUDA_VISIBLE_DEVICES`.
        gpu_energy_cost: Maps GPU indices to the electricity cost (in $) during the
            measurement window. GPU indices are from the DL framework's perspective
            after applying `CUDA_VISIBLE_DEVICES`.
        cpu_energy: Maps CPU indices to the energy consumed (in Joules) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d). This can
            be 'None' if CPU measurement is not available.
        cpu_energy_cost: Maps CPU indices to the electricity cost (in $) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d). This can
            be 'None' if CPU measurement is not available.
        dram_energy: Maps CPU indices to the energy consumed (in Joules) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d) and DRAM
            measurements are taken from sub-packages within each powerzone. This can be 'None' if
            CPU measurement is not available or DRAM measurement is not available.
        dram_energy_cost: Maps CPU indices to the electricity cost (in $) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d). This can be 'None' if
            CPU measurement is not available or DRAM measurement is not available.
    """

    time: float
    gpu_energy: dict[int, float]
    gpu_energy_cost: dict[int, float]
    cpu_energy: dict[int, float] | None = None
    cpu_energy_cost: dict[int, float] | None = None
    dram_energy: dict[int, float] | None = None
    dram_energy_cost: dict[int, float] | None = None


class Op(Enum):
    """Enum used to communicate between EnergyCostMonitor and _polling_process."""

    BEGIN = 0
    END = 1
    NEXTITER = 2


class EnergyCostMonitor:
    """Measure the energy, energy cost, and time consumption of a block of code.

    Works for multi-GPU and heterogeneous GPU types. Aware of `CUDA_VISIBLE_DEVICES`.
    For instance, if `CUDA_VISIBLE_DEVICES=2,3`, GPU index `1` passed into `gpu_indices`
    will be interpreted as CUDA device `3`.

    You can mark the beginning and end of a measurement window, during which the energy cost,
    GPU energy, and time consumed will be recorded. Multiple concurrent measurement windows
    are supported.
    """

    def __init__(
        self,
        electricity_price_provider: ElectricityPriceProvider,
        gpu_indices: list[int] | None = None,
        cpu_indices: list[int] | None = None,
        sync_execution_with: Literal["torch", "jax", "cupy"] = "torch",
    ) -> None:
        """Initializes Energy Cost Monitor.

        Args:
            electricity_price_provider: provider for which electricity price values will be fetched from
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
            approx_instant_energy=True,
        )
        self.electricity_price_provider = electricity_price_provider
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
                    self.electricity_price_provider,
                ),
            )
            self.polling_process.start()

        # start subwindows
        self.command_q.put((Op.BEGIN, key))

    def end_window(self, key: str, sync_execution: bool = True) -> EnergyCostMeasurement:
        """End a measurement window and return the time, energy consumption, and energy cost.

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
            gpu_energy_cost,
            cpu_energy_cost,
            dram_energy_cost,
        ) = self.finished_q.get()
        self.current_keys.remove(key)

        overall_measurement = self.zeus_monitor.end_window(key, sync_execution=sync_execution)

        measurement = EnergyCostMeasurement(
            time=overall_measurement.time,
            gpu_energy=overall_measurement.gpu_energy,
            cpu_energy=overall_measurement.cpu_energy,
            dram_energy=overall_measurement.dram_energy,
            gpu_energy_cost=gpu_energy_cost,
            cpu_energy_cost=cpu_energy_cost or None,
            dram_energy_cost=dram_energy_cost or None,
        )

        return measurement


def _polling_process(
    command_q: mp.Queue,
    finished_q: mp.Queue,
    gpu_indices: list[int],
    cpu_indices: list[int],
    electricity_price_provider: ElectricityPriceProvider,
):
    index = 0
    zeus_monitor = ZeusMonitor(
        gpu_indices=gpu_indices,
        cpu_indices=cpu_indices,
    )
    gpu_energy_cost = defaultdict(lambda: defaultdict(float))
    cpu_energy_cost = defaultdict(lambda: defaultdict(float))
    dram_energy_cost = defaultdict(lambda: defaultdict(float))
    energy_measurements = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    keys = set()

    # Fetch electricity price data
    try:
        electricity_price_data = electricity_price_provider.get_current_electricity_prices()
        energy_rate_structure = electricity_price_data["energy_rate_structure"]
        energy_weekday_schedule = electricity_price_data["energy_weekday_schedule"]
        energy_weekend_schedule = electricity_price_data["energy_weekend_schedule"]
    except Exception as e:
        logger.error("Failed to retrieve electricity price: %s.", e)
        return

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

    # update cumulative electricity costs
    def _update_energy_costs(key: str):
        for dt, measurement_map in energy_measurements[key].items():
            for index, energy in measurement_map.items():
                hardware_type, num_index = index.split("_")

                month = dt.month - 1
                hour = dt.hour
                day_of_week = dt.weekday()

                tier = energy_weekday_schedule[month][hour] if day_of_week < 5 else energy_weekend_schedule[month][hour]

                try:
                    flat_rate = energy_rate_structure[tier][0]["rate"]
                except (IndexError, KeyError, TypeError):
                    logger.error("Failed to parse electricity rate structure.")
                    return

                cost = (energy / 3.6e6) * flat_rate  # Convert Wh to kWh and multiply by rate

                if hardware_type == "gpu":
                    gpu_energy_cost[key][int(num_index)] += cost
                elif hardware_type == "cpu":
                    cpu_energy_cost[key][int(num_index)] += cost
                elif hardware_type == "dram":
                    dram_energy_cost[key][int(num_index)] += cost

        del energy_measurements[key]

    while True:
        # calculate current time
        now = datetime.now(timezone.utc)
        hour_floor = now.replace(minute=0, second=0, microsecond=0)
        hour_ceil = hour_floor + timedelta(hours=1)

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
                    _update_energy_costs(key)
                    finished_q.put(
                        (
                            dict(gpu_energy_cost.pop(key)),
                            dict(cpu_energy_cost.pop(key)),
                            dict(dram_energy_cost.pop(key)),
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
