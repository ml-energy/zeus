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
    with open(os.path.join(current_dir, "price_history_files", "invalid.json")) as f:
        INVALID_LAT_LON_RESPONSE = f.read()
    with open(os.path.join(current_dir, "price_history_files", "no_lat_lon.json")) as f:
        NO_LAT_LON_RESPONSE = f.read()
    with open(os.path.join(current_dir, "price_history_files", "ann_arbor.json")) as f:
        OPEN_EI_RESPONSE_ANN_ARBOR = f.read()
    with open(os.path.join(current_dir, "price_history_files", "virginia.json")) as f:
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
