from __future__ import annotations

from datetime import datetime, timezone
import json
import multiprocessing as mp
import dateutil
import pytest
import requests
import os

from unittest.mock import MagicMock, patch

from zeus.monitor.price import (
    ElectricityPriceProvider,
    OpenEIClient,
    Op,
    ZeusElectricityPriceNotFoundError,
    EnergyCostMonitor,
    _polling_process,
)

from zeus.utils.lat_lon import get_ip_lat_long


class MockHttpResponse:
    def __init__(self, text):
        self.text = text
        self.json_obj = json.loads(text)

    def json(self):
        return self.json_obj


@pytest.fixture
def mock_requests():
    IP_INFO_RESPONSE = """{
		"ip": "35.3.237.23",
		"hostname": "0587459863.wireless.umich.net",
		"city": "Ann Arbor",
		"region": "Michigan",
		"country": "US",
		"loc": "42.2776,-83.7409",
		"org": "AS36375 University of Michigan",
		"postal": "48109",
		"timezone": "America/Detroit",
		"readme": "https://ipinfo.io/missingauth"
	}"""

    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, "price_output_files", "invalid.json")) as f:
        INVALID_LAT_LON_RESPONSE = f.read()
    with open(os.path.join(current_dir, "price_output_files", "no_lat_lon.json")) as f:
        NO_LAT_LON_RESPONSE = f.read()
    with open(os.path.join(current_dir, "price_output_files", "ann_arbor.json")) as f:
        OPEN_EI_RESPONSE_ANN_ARBOR = f.read()
    with open(os.path.join(current_dir, "price_output_files", "virginia.json")) as f:
        OPEN_EI_RESPONSE_VIRGINIA = f.read()

    real_requests_get = requests.get

    def mock_requests_get(url, *args, **kwargs):
        if url == "http://ipinfo.io/json":
            return MockHttpResponse(IP_INFO_RESPONSE)
        elif url == (
            "https://api.openei.org/utility_rates?version=latest&format=json"
            + "&api_key=tJASWWgPhBRpiZCwfhtKV2A3gyNxbDfvQvdI5Wa7&lat=-100"
            + "&lon=-100&radius=0"
            + "&detail=full&sector=Residential"
        ):
            return MockHttpResponse(INVALID_LAT_LON_RESPONSE)
        elif url == (
            "https://api.openei.org/utility_rates?version=latest&format=json"
            + "&api_key=tJASWWgPhBRpiZCwfhtKV2A3gyNxbDfvQvdI5Wa7"
            + "&radius=0"
            + "&detail=full&sector=Residential"
        ):
            return MockHttpResponse(NO_LAT_LON_RESPONSE)
        elif url == (
            "https://api.openei.org/utility_rates?version=latest&format=json"
            + "&api_key=tJASWWgPhBRpiZCwfhtKV2A3gyNxbDfvQvdI5Wa7&lat=42.2776"
            + "&lon=-83.7409&radius=0"
            + "&detail=full&sector=Residential"
        ):
            return MockHttpResponse(OPEN_EI_RESPONSE_ANN_ARBOR)
        elif url == (
            "https://api.openei.org/utility_rates?version=latest&format=json"
            + "&api_key=tJASWWgPhBRpiZCwfhtKV2A3gyNxbDfvQvdI5Wa7&lat=38"
            + "&lon=-78&radius=0"
            + "&detail=full&sector=Residential"
        ):
            return MockHttpResponse(OPEN_EI_RESPONSE_VIRGINIA)
        else:
            return real_requests_get(url, *args, **kwargs)

    patch_request_get = patch("requests.get", side_effect=mock_requests_get)

    patch_request_get.start()

    yield

    patch_request_get.stop()


def test_get_prices(mock_requests):
    latlong = get_ip_lat_long()
    label = "539f6d3bec4f024411ecb311"
    assert latlong == (pytest.approx(42.2776), pytest.approx(-83.7409))
    client = OpenEIClient(latlong, label)

    prices = client.get_current_electricity_prices()
    assert prices["energy_rate_structure"] == [[{"rate": 0.1056, "adj": 0.029681}]]
    assert prices["energy_weekday_schedule"] == [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert prices["energy_weekend_schedule"] == [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]


def test_get_prices_invalid_lat_lon_no_response(mock_requests):
    latlong = [-100, -100]
    label = "539f6d3bec4f024411ecb311"
    client = OpenEIClient(latlong, label)

    with pytest.raises(ZeusElectricityPriceNotFoundError):
        client.get_current_electricity_prices()


def test_get_prices_invalid_label_no_response(mock_requests):
    latlong = get_ip_lat_long()
    assert latlong == (pytest.approx(42.2776), pytest.approx(-83.7409))
    label = "invalid"
    client = OpenEIClient(latlong, label)

    with pytest.raises(ZeusElectricityPriceNotFoundError):
        client.get_current_electricity_prices()


@pytest.fixture
def mock_zeus_monitor():
    patch_zeus_monitor = patch("zeus.monitor.price.ZeusMonitor", autospec=True)
    zeus_monitor = patch_zeus_monitor.start()

    # Configure the mock
    mock_instance = zeus_monitor.return_value
    mock_instance.end_window.return_value = MagicMock(
        gpu_energy={0: 30.0, 1: 35.0, 2: 40.0},
        cpu_energy={0: 20.0, 1: 25.0},
        dram_energy={0: 15.0, 1: 20.0},
    )
    mock_instance.gpu_indices = [0, 1, 2]
    mock_instance.cpu_indices = [0, 1]

    yield

    patch_zeus_monitor.stop()


class MockDateTime(datetime):
    times = []
    i = 0

    @classmethod
    def now(cls, tz=None):
        old_index = cls.i
        cls.i += 1
        return cls.times[old_index]

    @classmethod
    def get_previous_hour(cls):
        return cls.times[
            cls.i - 1
        ]  # minus one to handle off by one error: want to get previous hour


@pytest.fixture
def mock_datetime():
    patch_datetime_now = patch("zeus.monitor.carbon.datetime", new=MockDateTime)

    real_parse = dateutil.parser.parse

    def mock_parse(str_dt):
        dt = real_parse(str_dt)
        return MockDateTime(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
            tzinfo=timezone.utc,
        )

    patch_dateutil_parser = patch(
        "zeus.monitor.carbon.parser.parse", side_effect=mock_parse
    )
    patch_datetime_now.start()
    patch_dateutil_parser.start()

    yield

    patch_datetime_now.stop()
    patch_dateutil_parser.stop()


def get_expected_cpu_gpu_energy_costs(datetimes, label):
    expected_gpu_values = [0, 0, 0]
    expected_cpu_values = [0, 0]
    expected_dram_values = [0, 0]

    # Store mapping from stringified time to original floored datetime
    unique_datetimes = {}
    for dt in datetimes:
        floored = dt.replace(minute=0, second=0, microsecond=0)
        converted = floored.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        unique_datetimes[converted] = floored

    gpu_energy = [30, 35, 40]
    cpu_energy = [20, 25]
    dram_energy = [15, 20]

    with open("price_output_files/ann_arbor.json", "r") as file1:
        data = json.loads(file1.read())

        def search_json(data, key_name, target_value, return_value):
            results = []
            if isinstance(data, dict):
                for key, val in data.items():
                    if key == key_name and val == target_value:
                        results.append(data.get(return_value))
                    results.extend(
                        search_json(val, key_name, target_value, return_value)
                    )
            elif isinstance(data, list):
                for item in data:
                    results.extend(
                        search_json(item, key_name, target_value, return_value)
                    )
            return results

        def safe_search(label, key):
            result = search_json(data, "label", label, key)
            if not result or result[0] is None:
                raise ValueError(f"Missing '{key}' for label '{label}'")
            return result[0]

        energy_rate_structure = safe_search(label, "energyratestructure")
        energy_weekday_schedule = safe_search(label, "energyweekdayschedule")
        energy_weekend_schedule = safe_search(label, "energyweekendschedule")

        for time_str, time_obj in unique_datetimes.items():
            month = time_obj.month - 1
            hour = time_obj.hour
            day = time_obj.weekday()

            rate_index = (
                energy_weekend_schedule[month][hour]
                if day >= 5
                else energy_weekday_schedule[month][hour]
            )

            rate = energy_rate_structure[rate_index][0]["rate"]

            for i, energy in enumerate(gpu_energy):
                expected_gpu_values[i] += energy / 3.6e6 * rate
            for i, energy in enumerate(cpu_energy):
                expected_cpu_values[i] += energy / 3.6e6 * rate
            for i, energy in enumerate(dram_energy):
                expected_dram_values[i] += energy / 3.6e6 * rate

    return expected_gpu_values, expected_cpu_values, expected_dram_values


# test single window active for a window length of less than an one hour
def test_single_window_one_hour(mock_zeus_monitor, mock_requests, mock_datetime):
    command_q = mp.Queue()
    finished_q = mp.Queue()
    gpu_indices = [0, 1, 2]
    cpu_indices = [0, 1]
    label = "539f6d3bec4f024411ecb311"
    client = OpenEIClient((42.2776, -83.7409), label)

    MockDateTime.times = [
        MockDateTime(2025, 4, 1, 5, 30, tzinfo=timezone.utc),  # test_window start
        MockDateTime(2025, 4, 1, 5, 45, tzinfo=timezone.utc),  # test_window end
    ]
    MockDateTime.i = 0
    command_q.put((Op.BEGIN, "test_window"))  # (op, key)
    command_q.put((Op.END, "test_window"))

    _polling_process(command_q, finished_q, gpu_indices, cpu_indices, client)

    gpu_carbon_emission, cpu_carbon_emission, dram_carbon_emission = finished_q.get()

    # expected_values
    (
        expected_gpu_values,
        expected_cpu_values,
        expected_dram_values,
    ) = get_expected_cpu_gpu_energy_costs(MockDateTime.times, label)

    assert gpu_carbon_emission[0] == pytest.approx(expected_gpu_values[0])
    assert gpu_carbon_emission[1] == pytest.approx(expected_gpu_values[1])
    assert gpu_carbon_emission[2] == pytest.approx(expected_gpu_values[2])
    assert cpu_carbon_emission is not None
    assert cpu_carbon_emission[0] == pytest.approx(expected_cpu_values[0])
    assert cpu_carbon_emission[1] == pytest.approx(expected_cpu_values[1])
    assert dram_carbon_emission is not None
    assert dram_carbon_emission[0] == pytest.approx(expected_dram_values[0])
    assert dram_carbon_emission[1] == pytest.approx(expected_dram_values[1])
