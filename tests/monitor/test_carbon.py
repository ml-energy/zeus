from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import multiprocessing as mp
import dateutil
import pandas as pd
import pytest
import requests

from unittest.mock import MagicMock, patch

from zeus.monitor.carbon import (
    ElectrictyMapsClient,
    Op,
    TraceCarbonIntensityProvider,
    ZeusCarbonIntensityNotFoundError,
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

    NO_MEASUREMENT_RESPONSE = r'{"error":"No recent data for zone \"US-MIDW-MISO\""}'

    ELECTRICITY_MAPS_RESPONSE_LIFECYCLE = (
        '{"zone":"US-MIDW-MISO","carbonIntensity":466,"datetime":"2024-09-24T03:00:00.000Z",'
        '"updatedAt":"2024-09-24T02:47:02.408Z","createdAt":"2024-09-21T03:45:20.860Z",'
        '"emissionFactorType":"lifecycle","isEstimated":true,"estimationMethod":"TIME_SLICER_AVERAGE"}'
    )

    ELECTRICITY_MAPS_RESPONSE_DIRECT = (
        '{"zone":"US-MIDW-MISO","carbonIntensity":506,"datetime":"2024-09-27T00:00:00.000Z",'
        '"updatedAt":"2024-09-27T00:43:50.277Z","createdAt":"2024-09-24T00:46:38.741Z",'
        '"emissionFactorType":"direct","isEstimated":true,"estimationMethod":"TIME_SLICER_AVERAGE"}'
    )

    real_requests_get = requests.get

    def mock_requests_get(url, *args, **kwargs):
        if url == "http://ipinfo.io/json":
            return MockHttpResponse(IP_INFO_RESPONSE)
        elif (
            url
            == "https://api.electricitymap.org/v3/carbon-intensity/latest?lat=42.2776&lon=-83.7409&disableEstimations=True&emissionFactorType=direct"
        ):
            return MockHttpResponse(NO_MEASUREMENT_RESPONSE)
        elif (
            url
            == "https://api.electricitymap.org/v3/carbon-intensity/latest?lat=42.2776&lon=-83.7409&disableEstimations=False&emissionFactorType=direct"
        ):
            return MockHttpResponse(ELECTRICITY_MAPS_RESPONSE_DIRECT)
        elif (
            url
            == "https://api.electricitymap.org/v3/carbon-intensity/latest?lat=42.2776&lon=-83.7409&disableEstimations=False&emissionFactorType=lifecycle"
        ):
            return MockHttpResponse(ELECTRICITY_MAPS_RESPONSE_LIFECYCLE)
        else:
            return real_requests_get(url, *args, **kwargs)

    patch_request_get = patch("requests.get", side_effect=mock_requests_get)

    patch_request_get.start()

    yield

    patch_request_get.stop()


def test_get_current_carbon_intensity(mock_requests):
    latlong = get_ip_lat_long()
    assert latlong == (pytest.approx(42.2776), pytest.approx(-83.7409))
    provider = ElectrictyMapsClient(latlong, estimate=True, emission_factor_type="lifecycle")
    assert provider.get_current_carbon_intensity() == 466

    provider.emission_factor_type = "direct"
    assert provider.get_current_carbon_intensity() == 506


def test_get_current_carbon_intensity_no_response(mock_requests):
    latlong = get_ip_lat_long()
    assert latlong == (pytest.approx(42.2776), pytest.approx(-83.7409))
    provider = ElectrictyMapsClient(latlong)

    with pytest.raises(ZeusCarbonIntensityNotFoundError):
        provider.get_current_carbon_intensity()


@pytest.fixture
def mock_zeus_monitor():
    patch_zeus_monitor = patch("zeus.monitor.carbon.ZeusMonitor", autospec=True)
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


@pytest.fixture
def mock_carbon_history():
    real_requests_get = requests.get

    def mock_requests_get(url, *args, **kwargs):
        current_time = MockDateTime.get_previous_hour()
        TIME_TO_SWITCH = MockDateTime(2024, 12, 1, 7, 0, tzinfo=timezone.utc)
        if (
            url
            == "https://api.electricitymap.org/v3/carbon-intensity/history?lat=42.2776&lon=-83.7409&disableEstimations=False&emissionFactorType=lifecycle"
            and current_time < TIME_TO_SWITCH
        ):
            with open("tests/monitor/carbon_history_files/carbon_history_file1", "r") as file:
                content = file.read()
            return MockHttpResponse(content)
        elif (
            url
            == "https://api.electricitymap.org/v3/carbon-intensity/history?lat=42.2776&lon=-83.7409&disableEstimations=False&emissionFactorType=lifecycle"
        ):
            with open("tests/monitor/carbon_history_files/carbon_history_file2", "r") as file:
                content = file.read()
            return MockHttpResponse(content)
        else:
            return real_requests_get(url)

    patch_request_get = patch("requests.get", side_effect=mock_requests_get)
    patch_request_get.start()

    yield

    patch_request_get.stop()


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
        return cls.times[cls.i - 1]  # minus one to handle off by one error: want to get previous hour


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

    patch_dateutil_parser = patch("zeus.monitor.carbon.parser.parse", side_effect=mock_parse)
    patch_datetime_now.start()
    patch_dateutil_parser.start()

    yield

    patch_datetime_now.stop()
    patch_dateutil_parser.stop()


def get_expected_cpu_gpu_carbon_emision(datetimes):
    expected_gpu_values = [0, 0, 0]
    expected_cpu_values = [0, 0]
    expected_dram_values = [0, 0]
    unique_datetimes = set()

    # converts datetime objects to the most recent whole hour and turn into string format used in json response
    def convert(now):
        # converts datetime objects into string format used in json response
        hour_floor = now.replace(minute=0, second=0, microsecond=0)
        return hour_floor.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    for datetime in datetimes:
        unique_datetimes.add(convert(datetime))

    with (
        open("tests/monitor/carbon_history_files/carbon_history_file1", "r") as file1,
        open("tests/monitor/carbon_history_files/carbon_history_file2", "r") as file2,
    ):
        content1 = json.loads(file1.read())
        content2 = json.loads(file2.read())
        for time in unique_datetimes:
            for measurement in content1["history"]:
                if measurement["datetime"] == time:
                    expected_gpu_values[0] += 30.0 / 3.6e6 * measurement["carbonIntensity"]
                    expected_gpu_values[1] += 35.0 / 3.6e6 * measurement["carbonIntensity"]
                    expected_gpu_values[2] += 40.0 / 3.6e6 * measurement["carbonIntensity"]
                    expected_cpu_values[0] += 20.0 / 3.6e6 * measurement["carbonIntensity"]
                    expected_cpu_values[1] += 25.0 / 3.6e6 * measurement["carbonIntensity"]
                    expected_dram_values[0] += 15.0 / 3.6e6 * measurement["carbonIntensity"]
                    expected_dram_values[1] += 20.0 / 3.6e6 * measurement["carbonIntensity"]

            for measurement in content2["history"]:
                if measurement["datetime"] == time:
                    expected_gpu_values[0] += 30.0 / 3.6e6 * measurement["carbonIntensity"]
                    expected_gpu_values[1] += 35.0 / 3.6e6 * measurement["carbonIntensity"]
                    expected_gpu_values[2] += 40.0 / 3.6e6 * measurement["carbonIntensity"]
                    expected_cpu_values[0] += 20.0 / 3.6e6 * measurement["carbonIntensity"]
                    expected_cpu_values[1] += 25.0 / 3.6e6 * measurement["carbonIntensity"]

                    expected_dram_values[0] += 15.0 / 3.6e6 * measurement["carbonIntensity"]
                    expected_dram_values[1] += 20.0 / 3.6e6 * measurement["carbonIntensity"]

    return expected_gpu_values, expected_cpu_values, expected_dram_values


# test single window active for a window length of less than an one hour
def test_single_window_one_hour(mock_zeus_monitor, mock_carbon_history, mock_datetime):
    command_q = mp.Queue()
    finished_q = mp.Queue()
    gpu_indices = [0, 1, 2]
    cpu_indices = [0, 1]
    provider = ElectrictyMapsClient((42.2776, -83.7409), estimate=True, emission_factor_type="lifecycle")

    MockDateTime.times = [
        MockDateTime(2024, 12, 1, 5, 30, tzinfo=timezone.utc),  # test_window start
        MockDateTime(2024, 12, 1, 5, 45, tzinfo=timezone.utc),  # test_window end
    ]
    MockDateTime.i = 0
    command_q.put((Op.BEGIN, "test_window"))  # (op, key)
    command_q.put((Op.END, "test_window"))

    _polling_process(command_q, finished_q, gpu_indices, cpu_indices, provider)

    gpu_carbon_emission, cpu_carbon_emission, dram_carbon_emission = finished_q.get()

    # expected_values
    (
        expected_gpu_values,
        expected_cpu_values,
        expected_dram_values,
    ) = get_expected_cpu_gpu_carbon_emision(MockDateTime.times)

    assert gpu_carbon_emission[0] == pytest.approx(expected_gpu_values[0])
    assert gpu_carbon_emission[1] == pytest.approx(expected_gpu_values[1])
    assert gpu_carbon_emission[2] == pytest.approx(expected_gpu_values[2])
    assert cpu_carbon_emission is not None
    assert cpu_carbon_emission[0] == pytest.approx(expected_cpu_values[0])
    assert cpu_carbon_emission[1] == pytest.approx(expected_cpu_values[1])
    assert dram_carbon_emission is not None
    assert dram_carbon_emission[0] == pytest.approx(expected_dram_values[0])
    assert dram_carbon_emission[1] == pytest.approx(expected_dram_values[1])


# test single window active for a window length of at least 24 hours
def test_single_window_one_day(mock_zeus_monitor, mock_carbon_history, mock_datetime):
    command_q = mp.Queue()
    finished_q = mp.Queue()
    gpu_indices = [0, 1, 2]
    cpu_indices = [0, 1]
    provider = ElectrictyMapsClient((42.2776, -83.7409), estimate=True, emission_factor_type="lifecycle")

    # times so that exactly 25 iterations inside the polling loop executes
    MockDateTime.times = [
        MockDateTime(2024, 11, 30, 7, 0, tzinfo=timezone.utc),  # test_window start
        MockDateTime(2024, 11, 30, 8, 0, tzinfo=timezone.utc),  # extra datetime called after "start" is called
        MockDateTime(2024, 11, 30, 8, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 9, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 10, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 11, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 12, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 13, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 14, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 15, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 16, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 17, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 18, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 19, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 20, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 21, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 22, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 23, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 12, 1, 0, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 12, 1, 1, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 12, 1, 2, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 12, 1, 3, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 12, 1, 4, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 12, 1, 5, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 12, 1, 6, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 12, 1, 7, 0, tzinfo=timezone.utc),  # test_window end
    ]
    MockDateTime.i = 0

    command_q.put((Op.BEGIN, "test_window"))
    # polling process always calls get for each iteration
    # add nextiter ops to fast forward the polling process
    for i in range(23):
        command_q.put((Op.NEXTITER, None))
    command_q.put((Op.END, "test_window"))

    _polling_process(command_q, finished_q, gpu_indices, cpu_indices, provider)

    gpu_carbon_emission, cpu_carbon_emission, dram_carbon_emission = finished_q.get()

    (
        expected_gpu_values,
        expected_cpu_values,
        expected_dram_values,
    ) = get_expected_cpu_gpu_carbon_emision(MockDateTime.times)

    assert gpu_carbon_emission[0] == pytest.approx(expected_gpu_values[0])
    assert gpu_carbon_emission[1] == pytest.approx(expected_gpu_values[1])
    assert gpu_carbon_emission[2] == pytest.approx(expected_gpu_values[2])
    assert cpu_carbon_emission is not None
    assert cpu_carbon_emission[0] == pytest.approx(expected_cpu_values[0])
    assert cpu_carbon_emission[1] == pytest.approx(expected_cpu_values[1])
    assert dram_carbon_emission is not None
    assert dram_carbon_emission[0] == pytest.approx(expected_dram_values[0])
    assert dram_carbon_emission[1] == pytest.approx(expected_dram_values[1])


# test multiple windows active for a window length of less than an one hour
def test_multiple_windows_one_hour(mock_zeus_monitor, mock_carbon_history, mock_datetime):
    command_q = mp.Queue()
    finished_q = mp.Queue()
    gpu_indices = [0, 1, 2]
    cpu_indices = [0, 1]
    provider = ElectrictyMapsClient((42.2776, -83.7409), estimate=True, emission_factor_type="lifecycle")

    MockDateTime.times = [
        MockDateTime(2024, 12, 1, 4, 30, tzinfo=timezone.utc),  # test_window1 start
        MockDateTime(2024, 12, 1, 5, 0, tzinfo=timezone.utc),  # extra datetime called after "start" is called
        MockDateTime(2024, 12, 1, 5, 0, tzinfo=timezone.utc),  # test_window2 start
        MockDateTime(2024, 12, 1, 5, 30, tzinfo=timezone.utc),  # test_window1 end
        MockDateTime(2024, 12, 1, 5, 30, tzinfo=timezone.utc),  # test_window2 end
    ]
    MockDateTime.i = 0
    command_q.put((Op.BEGIN, "test_window1"))  # (op, key)
    command_q.put((Op.BEGIN, "test_window2"))  # (op, key)
    command_q.put((Op.END, "test_window1"))
    command_q.put((Op.END, "test_window2"))

    _polling_process(command_q, finished_q, gpu_indices, cpu_indices, provider)

    # retrieve values
    gpu_carbon_emission1, cpu_carbon_emission1, dram_carbon_emission1 = finished_q.get()
    gpu_carbon_emission2, cpu_carbon_emission2, dram_carbon_emission2 = finished_q.get()

    # expected_values
    (
        expected_gpu_values1,
        expected_cpu_values1,
        expected_dram_values1,
    ) = get_expected_cpu_gpu_carbon_emision(MockDateTime.times[:4])
    (
        expected_gpu_values2,
        expected_cpu_values2,
        expected_dram_values2,
    ) = get_expected_cpu_gpu_carbon_emision(MockDateTime.times[2:])

    # assert statements for test_window1
    assert gpu_carbon_emission1[0] == pytest.approx(expected_gpu_values1[0])
    assert gpu_carbon_emission1[1] == pytest.approx(expected_gpu_values1[1])
    assert gpu_carbon_emission1[2] == pytest.approx(expected_gpu_values1[2])
    assert cpu_carbon_emission1 is not None
    assert cpu_carbon_emission1[0] == pytest.approx(expected_cpu_values1[0])
    assert cpu_carbon_emission1[1] == pytest.approx(expected_cpu_values1[1])
    assert dram_carbon_emission1 is not None
    assert dram_carbon_emission1[0] == pytest.approx(expected_dram_values1[0])
    assert dram_carbon_emission1[1] == pytest.approx(expected_dram_values1[1])

    # assert statements for test_window2
    assert gpu_carbon_emission2[0] == pytest.approx(expected_gpu_values2[0])
    assert gpu_carbon_emission2[1] == pytest.approx(expected_gpu_values2[1])
    assert gpu_carbon_emission2[2] == pytest.approx(expected_gpu_values2[2])
    assert cpu_carbon_emission2 is not None
    assert cpu_carbon_emission2[0] == pytest.approx(expected_cpu_values2[0])
    assert cpu_carbon_emission2[1] == pytest.approx(expected_cpu_values2[1])
    assert dram_carbon_emission2 is not None
    assert dram_carbon_emission2[0] == pytest.approx(expected_dram_values2[0])
    assert dram_carbon_emission2[1] == pytest.approx(expected_dram_values2[1])


# test multiple windows active for a window length of at least a day
def test_multiple_windows_one_day(mock_zeus_monitor, mock_carbon_history, mock_datetime):
    command_q = mp.Queue()
    finished_q = mp.Queue()
    gpu_indices = [0, 1, 2]
    cpu_indices = [0, 1]
    provider = ElectrictyMapsClient((42.2776, -83.7409), estimate=True, emission_factor_type="lifecycle")

    MockDateTime.times = [
        MockDateTime(2024, 11, 30, 7, 0, tzinfo=timezone.utc),  # test_window1 start
        MockDateTime(2024, 11, 30, 8, 0, tzinfo=timezone.utc),  # extra datetime called after "start" is called
        MockDateTime(2024, 11, 30, 8, 0, tzinfo=timezone.utc),  # test_window2 start
        MockDateTime(2024, 11, 30, 9, 0, tzinfo=timezone.utc),  # extra datetime called after "start" is called
        MockDateTime(2024, 11, 30, 9, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 10, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 11, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 12, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 13, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 14, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 15, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 16, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 17, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 18, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 19, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 20, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 21, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 22, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 11, 30, 23, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 12, 1, 0, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 12, 1, 1, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 12, 1, 2, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 12, 1, 3, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 12, 1, 4, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 12, 1, 5, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 12, 1, 6, 0, tzinfo=timezone.utc),
        MockDateTime(2024, 12, 1, 7, 0, tzinfo=timezone.utc),  # test_window1 end
        MockDateTime(2024, 12, 1, 8, 0, tzinfo=timezone.utc),  # extra datetime called after "end" is called
        MockDateTime(2024, 12, 1, 8, 0, tzinfo=timezone.utc),  # test_window2 end
    ]
    MockDateTime.i = 0
    command_q.put((Op.BEGIN, "test_window1"))
    command_q.put((Op.BEGIN, "test_window2"))
    for i in range(22):
        command_q.put((Op.NEXTITER, None))
    command_q.put((Op.END, "test_window1"))
    command_q.put((Op.END, "test_window2"))

    _polling_process(command_q, finished_q, gpu_indices, cpu_indices, provider)

    # retrieve values
    gpu_carbon_emission1, cpu_carbon_emission1, dram_carbon_emission1 = finished_q.get()
    gpu_carbon_emission2, cpu_carbon_emission2, dram_carbon_emission2 = finished_q.get()

    # expected_values
    (
        expected_gpu_values1,
        expected_cpu_values1,
        expected_dram_values1,
    ) = get_expected_cpu_gpu_carbon_emision(MockDateTime.times[:27])
    (
        expected_gpu_values2,
        expected_cpu_values2,
        expected_dram_values2,
    ) = get_expected_cpu_gpu_carbon_emision(MockDateTime.times[2:])

    # assert statements for test_window1
    assert gpu_carbon_emission1[0] == pytest.approx(expected_gpu_values1[0])
    assert gpu_carbon_emission1[1] == pytest.approx(expected_gpu_values1[1])
    assert gpu_carbon_emission1[2] == pytest.approx(expected_gpu_values1[2])
    assert cpu_carbon_emission1 is not None
    assert cpu_carbon_emission1[0] == pytest.approx(expected_cpu_values1[0])
    assert cpu_carbon_emission1[1] == pytest.approx(expected_cpu_values1[1])
    assert dram_carbon_emission1 is not None
    assert dram_carbon_emission1[0] == pytest.approx(expected_dram_values1[0])
    assert dram_carbon_emission1[1] == pytest.approx(expected_dram_values1[1])

    # assert statements for test_window2
    assert gpu_carbon_emission2[0] == pytest.approx(expected_gpu_values2[0])
    assert gpu_carbon_emission2[1] == pytest.approx(expected_gpu_values2[1])
    assert gpu_carbon_emission2[2] == pytest.approx(expected_gpu_values2[2])
    assert cpu_carbon_emission2 is not None
    assert cpu_carbon_emission2[0] == pytest.approx(expected_cpu_values2[0])
    assert cpu_carbon_emission2[1] == pytest.approx(expected_cpu_values2[1])
    assert dram_carbon_emission2 is not None
    assert dram_carbon_emission2[0] == pytest.approx(expected_dram_values2[0])
    assert dram_carbon_emission2[1] == pytest.approx(expected_dram_values2[1])


UTC = timezone.utc
CENTRAL_DAYLIGHT = timezone(timedelta(hours=-5))
HOUR = timedelta(hours=1)
HALF_HOUR = timedelta(minutes=30)


def make_trace() -> dict[datetime, float]:
    return {
        datetime(2024, 7, 19, 0, tzinfo=UTC): 400.0,
        datetime(2024, 7, 19, 1, tzinfo=UTC): 300.0,
        datetime(2024, 7, 19, 2, tzinfo=UTC): 200.0,
    }


def test_trace_provider_step_lookup():
    provider = TraceCarbonIntensityProvider(make_trace(), period=HOUR, data_spacing=HOUR)
    assert provider.carbon_intensity_at(datetime(2024, 7, 19, 0, 0, tzinfo=UTC)) == 400.0
    assert provider.carbon_intensity_at(datetime(2024, 7, 19, 0, 59, 59, tzinfo=UTC)) == 400.0
    assert provider.carbon_intensity_at(datetime(2024, 7, 19, 1, 0, tzinfo=UTC)) == 300.0
    # 20:30 on Jul 18 in Central Daylight Time is 01:30 UTC on Jul 19.
    assert provider.carbon_intensity_at(datetime(2024, 7, 18, 20, 30, tzinfo=CENTRAL_DAYLIGHT)) == 300.0
    assert provider.update_period == timedelta(hours=1)
    assert provider.history_length == 3


def test_trace_provider_out_of_range(caplog):
    provider = TraceCarbonIntensityProvider(make_trace(), period=HOUR, data_spacing=HOUR)
    # The trace is from 2024, so it has no value for now.
    assert "no value for the current period" in caplog.text
    with pytest.raises(ZeusCarbonIntensityNotFoundError, match="replay_start"):
        provider.carbon_intensity_at(datetime(2024, 7, 19, 3, 0, tzinfo=UTC))
    with pytest.raises(ZeusCarbonIntensityNotFoundError):
        provider.carbon_intensity_at(datetime(2024, 7, 18, 23, 59, tzinfo=UTC))
    with pytest.raises(ValueError):
        provider.carbon_intensity_at(datetime(2024, 7, 19, 1, 0))


@pytest.mark.parametrize(
    "trace",
    [
        {},
        {datetime(2024, 7, 19, 0): 400.0},
        {datetime(2024, 7, 19, 0, 30, tzinfo=UTC): 400.0},
        {datetime(2024, 7, 19, 0, tzinfo=UTC): -1.0},
        {datetime(2024, 7, 19, 0, tzinfo=UTC): float("nan")},
    ],
)
def test_trace_provider_rejects_bad_traces(trace):
    with pytest.raises(ValueError):
        TraceCarbonIntensityProvider(trace, period=HOUR, data_spacing=HOUR)


def test_trace_provider_period_must_fit_the_data():
    with pytest.raises(ValueError, match="positive"):
        TraceCarbonIntensityProvider(make_trace(), period=timedelta(0), data_spacing=HOUR)
    # Hourly data cannot be split into 30-minute periods.
    with pytest.raises(ValueError, match="coarser"):
        TraceCarbonIntensityProvider(make_trace(), period=HALF_HOUR, data_spacing=HOUR)
    # 45-minute periods are not a whole number of 30-minute data points.
    half_hourly = {datetime(2024, 7, 19, 0, 0, tzinfo=UTC): 1.0, datetime(2024, 7, 19, 0, 30, tzinfo=UTC): 2.0}
    with pytest.raises(ValueError, match="whole multiple"):
        TraceCarbonIntensityProvider(half_hourly, period=timedelta(minutes=45), data_spacing=HALF_HOUR)
    # A stray half-hour point in hourly data raises.
    stray = {datetime(2024, 7, 19, hour, tzinfo=UTC): 100.0 for hour in range(6)}
    stray[datetime(2024, 7, 19, 2, 30, tzinfo=UTC)] = 250.0
    with pytest.raises(ValueError, match="not on the"):
        TraceCarbonIntensityProvider(stray, period=HOUR, data_spacing=HOUR)


def test_trace_provider_data_spacing():
    # Hourly data with 01:00 missing. With the spacing given, the gap is just a missing hour.
    sparse = {datetime(2024, 7, 19, 0, tzinfo=UTC): 400.0, datetime(2024, 7, 19, 2, tzinfo=UTC): 200.0}
    provider = TraceCarbonIntensityProvider(sparse, period=HOUR, data_spacing=HOUR)
    assert provider.history_length == 2
    with pytest.raises(ZeusCarbonIntensityNotFoundError):
        provider.carbon_intensity_at(datetime(2024, 7, 19, 1, 30, tzinfo=UTC))
    single = {datetime(2024, 7, 19, 0, tzinfo=UTC): 400.0}
    assert TraceCarbonIntensityProvider(single, period=HOUR, data_spacing=HOUR).history_length == 1
    with pytest.raises(ValueError, match="positive"):
        TraceCarbonIntensityProvider(sparse, period=HOUR, data_spacing=timedelta(0))
    # Hourly data passed as 30-minute data fills only half of each hour.
    with pytest.raises(ValueError, match="fully covered"):
        TraceCarbonIntensityProvider(make_trace(), period=HOUR, data_spacing=HALF_HOUR)


def test_trace_provider_averages_finer_data(caplog):
    # 15-minute data averaged into hourly periods.
    trace = {
        datetime(2024, 7, 19, 0, minute, tzinfo=UTC): value
        for minute, value in zip((0, 15, 30, 45), (100.0, 200.0, 300.0, 400.0), strict=True)
    }
    trace[datetime(2024, 7, 19, 1, 0, tzinfo=UTC)] = 500.0  # only one of the four points in this hour
    provider = TraceCarbonIntensityProvider(trace, period=HOUR, data_spacing=timedelta(minutes=15))
    assert provider.update_period == HOUR
    assert provider.history_length == 1
    assert "Left out 1 of 2 periods" in caplog.text
    assert provider.carbon_intensity_at(datetime(2024, 7, 19, 0, 50, tzinfo=UTC)) == 250.0
    with pytest.raises(ZeusCarbonIntensityNotFoundError):  # 01:00 is not fully covered, so it is left out
        provider.carbon_intensity_at(datetime(2024, 7, 19, 1, 10, tzinfo=UTC))

    partial = {datetime(2024, 7, 19, 0, 0, tzinfo=UTC): 1.0, datetime(2024, 7, 19, 0, 15, tzinfo=UTC): 2.0}
    with pytest.raises(ValueError, match="fully covered"):
        TraceCarbonIntensityProvider(partial, period=HOUR, data_spacing=timedelta(minutes=15))


def test_trace_provider_from_csv_hour_ending(tmp_path):
    # EIA-930 style: naive UTC timestamps that mark the end of each hour, intensity in lb/kWh.
    path = tmp_path / "eia930.csv"
    path.write_text("utc_end,ci_lb_kwh\n2024-07-19T01:00:00,0.5\n2024-07-19T02:00:00,0.25\n")
    provider = TraceCarbonIntensityProvider.from_csv(
        path,
        "utc_end",
        "ci_lb_kwh",
        period=HOUR,
        data_spacing=HOUR,
        time_zone="UTC",
        time_label="end",
        scale=453.59237,
    )
    assert provider.carbon_intensity_at(datetime(2024, 7, 19, 0, 30, tzinfo=UTC)) == pytest.approx(226.796185)
    assert provider.carbon_intensity_at(datetime(2024, 7, 19, 1, 30, tzinfo=UTC)) == pytest.approx(113.39809)
    with pytest.raises(ZeusCarbonIntensityNotFoundError):
        provider.carbon_intensity_at(datetime(2024, 7, 19, 2, 0, tzinfo=UTC))


def test_trace_provider_from_csv_offsets(tmp_path):
    # Local times with offsets that change across the start of daylight saving time.
    path = tmp_path / "local.csv"
    path.write_text("time,ci\n2026-03-08 01:00:00-0600,300\n2026-03-08 03:00:00-0500,310\n")
    provider = TraceCarbonIntensityProvider.from_csv(path, "time", "ci", period=HOUR, data_spacing=HOUR)
    assert provider.carbon_intensity_at(datetime(2026, 3, 8, 7, 0, tzinfo=UTC)) == 300.0
    assert provider.carbon_intensity_at(datetime(2026, 3, 8, 8, 0, tzinfo=UTC)) == 310.0
    with pytest.raises(ValueError, match="already has UTC offsets"):
        TraceCarbonIntensityProvider.from_csv(path, "time", "ci", period=HOUR, data_spacing=HOUR, time_zone="UTC")

    # PostgreSQL writes offsets as +00.
    postgres = tmp_path / "postgres.csv"
    postgres.write_text("time,ci\n2026-03-08 07:00:00+00,300\n2026-03-08 08:00:00+00,310\n")
    provider = TraceCarbonIntensityProvider.from_csv(postgres, "time", "ci", period=HOUR, data_spacing=HOUR)
    assert provider.carbon_intensity_at(datetime(2026, 3, 8, 8, 0, tzinfo=UTC)) == 310.0

    # Rows that differ only in precision, with and without offsets.
    precision = tmp_path / "precision.csv"
    precision.write_text("time,ci\n2024-07-19T00:00:00Z,300\n2024-07-19T01:00:00.000Z,310\n")
    provider = TraceCarbonIntensityProvider.from_csv(precision, "time", "ci", period=HOUR, data_spacing=HOUR)
    assert provider.carbon_intensity_at(datetime(2024, 7, 19, 1, 0, tzinfo=UTC)) == 310.0
    precision.write_text("time,ci\n2024-07-19 00:00:00,300\n2024-07-19 01:00:00.000,310\n")
    provider = TraceCarbonIntensityProvider.from_csv(
        precision, "time", "ci", period=HOUR, data_spacing=HOUR, time_zone="UTC"
    )
    assert provider.carbon_intensity_at(datetime(2024, 7, 19, 1, 0, tzinfo=UTC)) == 310.0

    naive = tmp_path / "naive.csv"
    naive.write_text("time,ci\n2026-03-08 01:00:00,300\n")
    with pytest.raises(ValueError, match="Pass `time_zone`"):
        TraceCarbonIntensityProvider.from_csv(naive, "time", "ci", period=HOUR, data_spacing=HOUR)

    mixed = tmp_path / "mixed.csv"
    mixed.write_text("time,ci\n2026-03-08 01:00:00-0600,300\n2026-03-08 02:00:00,310\n")
    with pytest.raises(ValueError, match="mixes"):
        TraceCarbonIntensityProvider.from_csv(mixed, "time", "ci", period=HOUR, data_spacing=HOUR)

    repeated = tmp_path / "repeated.csv"
    repeated.write_text("time,ci\n2026-03-08T01:00:00Z,300\n2026-03-08T01:00:00Z,310\n")
    with pytest.raises(ValueError, match="repeated"):
        TraceCarbonIntensityProvider.from_csv(repeated, "time", "ci", period=HOUR, data_spacing=HOUR)


def test_trace_provider_replay(mock_datetime):
    MockDateTime.times = [
        MockDateTime(2026, 11, 15, 10, 20, tzinfo=UTC),  # provider created
        MockDateTime(2026, 11, 15, 12, 5, tzinfo=UTC),  # get_current_carbon_intensity
    ]
    MockDateTime.i = 0
    provider = TraceCarbonIntensityProvider(
        make_trace(), period=HOUR, data_spacing=HOUR, replay_start=datetime(2024, 7, 19, 0, 45, tzinfo=UTC)
    )

    # 10:00 on the wall clock is replayed as 00:00 of the trace, so 12:05 reads the 02:00 value.
    assert provider.get_current_carbon_intensity() == 200.0
    assert provider.get_recent_carbon_intensity() == {
        datetime(2026, 11, 15, 10, tzinfo=UTC): 400.0,
        datetime(2026, 11, 15, 11, tzinfo=UTC): 300.0,
        datetime(2026, 11, 15, 12, tzinfo=UTC): 200.0,
    }

    with pytest.raises(ZeusCarbonIntensityNotFoundError) as error:
        provider.carbon_intensity_at(datetime(2024, 7, 19, 3, tzinfo=UTC))
    assert "replay_start" not in str(error.value)

    with pytest.raises(ValueError, match="not inside the trace"):
        TraceCarbonIntensityProvider(
            make_trace(), period=HOUR, data_spacing=HOUR, replay_start=datetime(2024, 7, 20, tzinfo=UTC)
        )


def test_trace_provider_in_polling_process(mock_zeus_monitor, mock_datetime):
    command_q = mp.Queue()
    finished_q = mp.Queue()
    trace = {
        datetime(2024, 11, 30, 7, tzinfo=UTC): 400.0,
        datetime(2024, 11, 30, 8, tzinfo=UTC): 300.0,
        datetime(2024, 11, 30, 9, tzinfo=UTC): 200.0,
    }
    MockDateTime.times = [MockDateTime(2024, 11, 30, 7, 0, tzinfo=UTC)]  # provider created
    MockDateTime.i = 0
    provider = TraceCarbonIntensityProvider(trace, period=HOUR, data_spacing=HOUR)

    # One window over three hours. The trace is refetched after the second hour.
    MockDateTime.times = [
        MockDateTime(2024, 11, 30, 7, 0, tzinfo=UTC),  # window start
        MockDateTime(2024, 11, 30, 8, 0, tzinfo=UTC),  # after BEGIN, crosses the hour
        MockDateTime(2024, 11, 30, 8, 0, tzinfo=UTC),
        MockDateTime(2024, 11, 30, 9, 0, tzinfo=UTC),  # window end
    ]
    MockDateTime.i = 0
    command_q.put((Op.BEGIN, "test_window"))
    command_q.put((Op.NEXTITER, None))
    command_q.put((Op.END, "test_window"))

    _polling_process(command_q, finished_q, [0, 1, 2], [0, 1], provider)
    gpu_carbon_emission, cpu_carbon_emission, dram_carbon_emission = finished_q.get()

    # The mocked monitor reports the same energy for every hour (GPU 0: 30 J, CPU 0: 20 J, DRAM 0: 15 J).
    total_intensity = 400.0 + 300.0 + 200.0
    assert gpu_carbon_emission[0] == pytest.approx(30.0 / 3.6e6 * total_intensity)
    assert gpu_carbon_emission[2] == pytest.approx(40.0 / 3.6e6 * total_intensity)
    assert cpu_carbon_emission[0] == pytest.approx(20.0 / 3.6e6 * total_intensity)
    assert dram_carbon_emission[0] == pytest.approx(15.0 / 3.6e6 * total_intensity)


def test_trace_provider_sub_hourly_period():
    trace = {
        datetime(2024, 7, 19, 0, 0, tzinfo=UTC): 112.0,
        datetime(2024, 7, 19, 0, 30, tzinfo=UTC): 108.0,
        datetime(2024, 7, 19, 1, 30, tzinfo=UTC): 106.0,  # 01:00 is a gap
    }
    provider = TraceCarbonIntensityProvider(trace, period=HALF_HOUR, data_spacing=HALF_HOUR)
    assert provider.update_period == HALF_HOUR
    assert provider.carbon_intensity_at(datetime(2024, 7, 19, 0, 29, tzinfo=UTC)) == 112.0
    assert provider.carbon_intensity_at(datetime(2024, 7, 19, 0, 45, tzinfo=UTC)) == 108.0
    with pytest.raises(ZeusCarbonIntensityNotFoundError):
        provider.carbon_intensity_at(datetime(2024, 7, 19, 1, 10, tzinfo=UTC))


def test_trace_provider_rejects_off_grid_times():
    # 30-minute spacing that does not start at :00 and :30.
    trace = {datetime(2024, 7, 19, 0, 10, tzinfo=UTC): 1.0, datetime(2024, 7, 19, 0, 40, tzinfo=UTC): 2.0}
    with pytest.raises(ValueError, match="not on the"):
        TraceCarbonIntensityProvider(trace, period=HOUR, data_spacing=HALF_HOUR)


def test_trace_provider_half_hour_periods_in_polling_process(mock_zeus_monitor, mock_datetime):
    command_q = mp.Queue()
    finished_q = mp.Queue()
    MockDateTime.times = [MockDateTime(2024, 11, 30, 7, 0, tzinfo=UTC)]  # provider created
    MockDateTime.i = 0
    provider = TraceCarbonIntensityProvider(
        {datetime(2024, 11, 30, 7, 0, tzinfo=UTC): 400.0, datetime(2024, 11, 30, 7, 30, tzinfo=UTC): 100.0},
        period=HALF_HOUR,
        data_spacing=HALF_HOUR,
    )

    # One window across a half-hour boundary: energy is grouped by 30-minute period.
    MockDateTime.times = [
        MockDateTime(2024, 11, 30, 7, 0, tzinfo=UTC),  # window start
        MockDateTime(2024, 11, 30, 7, 30, tzinfo=UTC),  # after BEGIN, crosses the half hour
        MockDateTime(2024, 11, 30, 7, 30, tzinfo=UTC),  # window end
    ]
    MockDateTime.i = 0
    command_q.put((Op.BEGIN, "test_window"))
    command_q.put((Op.END, "test_window"))

    _polling_process(command_q, finished_q, [0, 1, 2], [0, 1], provider)
    gpu_carbon_emission, _, _ = finished_q.get()

    # The mocked monitor reports 30 J for GPU 0 in each period.
    assert gpu_carbon_emission[0] == pytest.approx(30.0 / 3.6e6 * (400.0 + 100.0))


@pytest.mark.parametrize(
    "header",
    [
        # 2024 exports
        "Datetime (UTC),Country,Zone name,Zone id,Carbon intensity gCO₂eq/kWh (direct),"
        "Carbon intensity gCO₂eq/kWh (Life cycle),Carbon-free energy percentage (CFE%),"
        "Renewable energy percentage (RE%),Data source",
        # 2023 exports
        "Datetime (UTC),Country,Zone Name,Zone Id,Carbon Intensity gCO₂eq/kWh (direct),"
        "Carbon Intensity gCO₂eq/kWh (LCA),Low Carbon Percentage,Renewable Percentage,Data Source",
    ],
)
def test_trace_provider_from_electricitymaps_csv(tmp_path, header):
    path = tmp_path / "US-TEX-ERCO_2024_hourly.csv"
    path.write_text(
        header + "\n"
        "2024-07-19 00:00:00,USA,Texas,US-TEX-ERCO,300.5,350.5,40.0,30.0,eia.gov\n"
        "2024-07-19 01:00:00,USA,Texas,US-TEX-ERCO,310.5,360.5,40.0,30.0,eia.gov\n",
        encoding="utf-8",
    )
    direct = TraceCarbonIntensityProvider.from_electricitymaps_csv(path, period=HOUR, data_spacing=HOUR)
    lifecycle = TraceCarbonIntensityProvider.from_electricitymaps_csv(
        path, period=HOUR, data_spacing=HOUR, emission_factor_type="lifecycle"
    )
    assert lifecycle.carbon_intensity_at(datetime(2024, 7, 19, 1, 15, tzinfo=UTC)) == 360.5
    assert direct.carbon_intensity_at(datetime(2024, 7, 19, 0, 15, tzinfo=UTC)) == 300.5
    assert lifecycle.update_period == HOUR

    # An hour without a value is skipped. The file is still read as hourly.
    gap = tmp_path / "gap.csv"
    gap.write_text(
        header + "\n"
        "2024-07-19 00:00:00,USA,Texas,US-TEX-ERCO,400,450,40.0,30.0,eia.gov\n"
        "2024-07-19 01:00:00,USA,Texas,US-TEX-ERCO,,,40.0,30.0,eia.gov\n"
        "2024-07-19 02:00:00,USA,Texas,US-TEX-ERCO,200,250,40.0,30.0,eia.gov\n",
        encoding="utf-8",
    )
    provider = TraceCarbonIntensityProvider.from_electricitymaps_csv(gap, period=HOUR, data_spacing=HOUR)
    assert provider.carbon_intensity_at(datetime(2024, 7, 19, 0, 30, tzinfo=UTC)) == 400.0
    with pytest.raises(ZeusCarbonIntensityNotFoundError):
        provider.carbon_intensity_at(datetime(2024, 7, 19, 1, 30, tzinfo=UTC))

    other = tmp_path / "other.csv"
    other.write_text("time,ci\n2024-07-19 00:00:00,1\n")
    with pytest.raises(ValueError, match="Electricity Maps"):
        TraceCarbonIntensityProvider.from_electricitymaps_csv(other, period=HOUR, data_spacing=HOUR)


def test_trace_provider_from_eia930_xlsx(mocker):
    # The "Published Hourly Data" sheet: UTC time marks the end of each hour, intensity is in
    # lb/kWh, and hours before CO2 estimates start (July 2018) are empty.
    sheet = pd.DataFrame(
        {
            "UTC time": pd.to_datetime(["2018-07-01 05:00", "2018-07-01 06:00", "2018-07-01 07:00"]),
            "CO2 Emissions Intensity for Consumed Electricity": [float("nan"), 0.5, 0.25],
            "CO2 Emissions Intensity for Generated Electricity": [float("nan"), 0.4, 0.2],
        }
    )
    read_excel = mocker.patch("zeus.monitor.carbon.pd.read_excel", return_value=sheet)
    provider = TraceCarbonIntensityProvider.from_eia930_xlsx("ERCO.xlsx", period=HOUR)
    assert read_excel.call_args.kwargs["sheet_name"] == "Published Hourly Data"
    assert provider.history_length == 2
    assert provider.carbon_intensity_at(datetime(2018, 7, 1, 5, 30, tzinfo=UTC)) == pytest.approx(0.5 * 453.59237)
    assert provider.carbon_intensity_at(datetime(2018, 7, 1, 6, 30, tzinfo=UTC)) == pytest.approx(0.25 * 453.59237)
    with pytest.raises(ZeusCarbonIntensityNotFoundError):
        provider.carbon_intensity_at(datetime(2018, 7, 1, 4, 30, tzinfo=UTC))
    generated = TraceCarbonIntensityProvider.from_eia930_xlsx("ERCO.xlsx", period=HOUR, electricity="generated")
    assert generated.carbon_intensity_at(datetime(2018, 7, 1, 5, 30, tzinfo=UTC)) == pytest.approx(0.4 * 453.59237)
    # One row is one hour of data, so it cannot fill a two-hour period.
    read_excel.return_value = sheet.iloc[[1]]
    with pytest.raises(ValueError, match="fully covered"):
        TraceCarbonIntensityProvider.from_eia930_xlsx("ERCO.xlsx", period=2 * HOUR)


def test_trace_provider_from_gb_carbon_intensity_json(tmp_path):
    # A saved response from https://api.carbonintensity.org.uk/intensity/{from}/{to}
    path = tmp_path / "gb.json"
    path.write_text(
        '{"data":['
        '{"from":"2024-07-19T00:00Z","to":"2024-07-19T00:30Z","intensity":{"forecast":118,"actual":108,"index":"low"}},'
        '{"from":"2024-07-19T00:30Z","to":"2024-07-19T01:00Z","intensity":{"forecast":108,"actual":103,"index":"low"}},'
        '{"from":"2024-07-19T01:00Z","to":"2024-07-19T01:30Z","intensity":{"forecast":101,"actual":null,"index":"low"}}]}'
    )
    actual = TraceCarbonIntensityProvider.from_gb_carbon_intensity_json(path, period=HALF_HOUR)
    forecast = TraceCarbonIntensityProvider.from_gb_carbon_intensity_json(path, period=HALF_HOUR, value="forecast")
    assert actual.update_period == HALF_HOUR
    assert actual.carbon_intensity_at(datetime(2024, 7, 19, 0, 40, tzinfo=UTC)) == 103.0
    assert forecast.carbon_intensity_at(datetime(2024, 7, 19, 1, 10, tzinfo=UTC)) == 101.0
    with pytest.raises(ZeusCarbonIntensityNotFoundError):  # no actual value for 01:00
        actual.carbon_intensity_at(datetime(2024, 7, 19, 1, 10, tzinfo=UTC))

    # Hourly periods average the two half hours. The 01:00 hour has no actual value and no 01:30
    # interval, so it is left out.
    hourly = TraceCarbonIntensityProvider.from_gb_carbon_intensity_json(path, period=HOUR)
    assert hourly.history_length == 1
    assert hourly.carbon_intensity_at(datetime(2024, 7, 19, 0, 40, tzinfo=UTC)) == pytest.approx((108 + 103) / 2)

    # A single 30-minute interval cannot fill an hour.
    single = tmp_path / "single.json"
    single.write_text(
        '{"data":[{"from":"2024-07-19T00:00Z","to":"2024-07-19T00:30Z",'
        '"intensity":{"forecast":118,"actual":108,"index":"low"}}]}'
    )
    with pytest.raises(ValueError, match="fully covered"):
        TraceCarbonIntensityProvider.from_gb_carbon_intensity_json(single, period=HOUR)
