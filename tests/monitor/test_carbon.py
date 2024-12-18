from __future__ import annotations

from datetime import datetime, timezone
import json
import multiprocessing as mp
import dateutil
import pytest
import requests

from unittest.mock import MagicMock, patch

from zeus.monitor.carbon import (
    ElectrictyMapsClient,
    get_ip_lat_long,
    Op,
    ZeusCarbonIntensityNotFoundError,
    _polling_process,
)


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
    provider = ElectrictyMapsClient(
        latlong, estimate=True, emission_factor_type="lifecycle"
    )
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
            with open(
                "tests/monitor/carbon_history_files/carbon_history_file1", "r"
            ) as file:
                content = file.read()
            return MockHttpResponse(content)
        elif (
            url
            == "https://api.electricitymap.org/v3/carbon-intensity/history?lat=42.2776&lon=-83.7409&disableEstimations=False&emissionFactorType=lifecycle"
        ):
            with open(
                "tests/monitor/carbon_history_files/carbon_history_file2", "r"
            ) as file:
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
                    expected_gpu_values[0] += (
                        30.0 / 3.6e6 * measurement["carbonIntensity"]
                    )
                    expected_gpu_values[1] += (
                        35.0 / 3.6e6 * measurement["carbonIntensity"]
                    )
                    expected_gpu_values[2] += (
                        40.0 / 3.6e6 * measurement["carbonIntensity"]
                    )
                    expected_cpu_values[0] += (
                        20.0 / 3.6e6 * measurement["carbonIntensity"]
                    )
                    expected_cpu_values[1] += (
                        25.0 / 3.6e6 * measurement["carbonIntensity"]
                    )
                    expected_dram_values[0] += (
                        15.0 / 3.6e6 * measurement["carbonIntensity"]
                    )
                    expected_dram_values[1] += (
                        20.0 / 3.6e6 * measurement["carbonIntensity"]
                    )

            for measurement in content2["history"]:
                if measurement["datetime"] == time:
                    expected_gpu_values[0] += (
                        30.0 / 3.6e6 * measurement["carbonIntensity"]
                    )
                    expected_gpu_values[1] += (
                        35.0 / 3.6e6 * measurement["carbonIntensity"]
                    )
                    expected_gpu_values[2] += (
                        40.0 / 3.6e6 * measurement["carbonIntensity"]
                    )
                    expected_cpu_values[0] += (
                        20.0 / 3.6e6 * measurement["carbonIntensity"]
                    )
                    expected_cpu_values[1] += (
                        25.0 / 3.6e6 * measurement["carbonIntensity"]
                    )

                    expected_dram_values[0] += (
                        15.0 / 3.6e6 * measurement["carbonIntensity"]
                    )
                    expected_dram_values[1] += (
                        20.0 / 3.6e6 * measurement["carbonIntensity"]
                    )

    return expected_gpu_values, expected_cpu_values, expected_dram_values


# test single window active for a window length of less than an one hour
def test_single_window_one_hour(mock_zeus_monitor, mock_carbon_history, mock_datetime):
    command_q = mp.Queue()
    finished_q = mp.Queue()
    gpu_indices = [0, 1, 2]
    cpu_indices = [0, 1]
    provider = ElectrictyMapsClient(
        (42.2776, -83.7409), estimate=True, emission_factor_type="lifecycle"
    )

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
    provider = ElectrictyMapsClient(
        (42.2776, -83.7409), estimate=True, emission_factor_type="lifecycle"
    )

    # times so that exactly 25 iterations inside the polling loop executes
    MockDateTime.times = [
        MockDateTime(2024, 11, 30, 7, 0, tzinfo=timezone.utc),  # test_window start
        MockDateTime(
            2024, 11, 30, 8, 0, tzinfo=timezone.utc
        ),  # extra datetime called after "start" is called
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
def test_multiple_windows_one_hour(
    mock_zeus_monitor, mock_carbon_history, mock_datetime
):
    command_q = mp.Queue()
    finished_q = mp.Queue()
    gpu_indices = [0, 1, 2]
    cpu_indices = [0, 1]
    provider = ElectrictyMapsClient(
        (42.2776, -83.7409), estimate=True, emission_factor_type="lifecycle"
    )

    MockDateTime.times = [
        MockDateTime(2024, 12, 1, 4, 30, tzinfo=timezone.utc),  # test_window1 start
        MockDateTime(
            2024, 12, 1, 5, 0, tzinfo=timezone.utc
        ),  # extra datetime called after "start" is called
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
def test_multiple_windows_one_day(
    mock_zeus_monitor, mock_carbon_history, mock_datetime
):
    command_q = mp.Queue()
    finished_q = mp.Queue()
    gpu_indices = [0, 1, 2]
    cpu_indices = [0, 1]
    provider = ElectrictyMapsClient(
        (42.2776, -83.7409), estimate=True, emission_factor_type="lifecycle"
    )

    MockDateTime.times = [
        MockDateTime(2024, 11, 30, 7, 0, tzinfo=timezone.utc),  # test_window1 start
        MockDateTime(
            2024, 11, 30, 8, 0, tzinfo=timezone.utc
        ),  # extra datetime called after "start" is called
        MockDateTime(2024, 11, 30, 8, 0, tzinfo=timezone.utc),  # test_window2 start
        MockDateTime(
            2024, 11, 30, 9, 0, tzinfo=timezone.utc
        ),  # extra datetime called after "start" is called
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
        MockDateTime(
            2024, 12, 1, 8, 0, tzinfo=timezone.utc
        ),  # extra datetime called after "end" is called
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
