import requests
import pytest
import json

from unittest.mock import patch

from zeus.carbon import get_ip_lat_long
from zeus.carbon.electricity_maps_carbon_intensity_provider import (
    ElectricityMapsCarbonIntensityProvider,
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

    NO_MEASUREMENT_RESPONSE = '{"error":"No recent data for zone "US-MIDW-MISO""}'

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


@pytest.fixture
def mock_exception_ip():
    def mock_requests_get(url):
        raise ConnectionError

    patch_request_get = patch("requests.get", side_effect=mock_requests_get)

    patch_request_get.start()

    yield

    patch_request_get.stop()


def test_get_current_carbon_intensity(mock_requests):
    latlong = get_ip_lat_long()
    assert latlong == (42.2776, -83.7409)
    provider = ElectricityMapsCarbonIntensityProvider(latlong)
    assert (
        provider.get_current_carbon_intensity(
            estimate=True, emission_factor_type="lifecycle"
        )
        == 466
    )
    assert provider.get_current_carbon_intensity(estimate=True) == 506


def test_get_current_carbon_intensity_no_response(mock_requests):
    latlong = get_ip_lat_long()
    assert latlong == (42.2776, -83.7409)
    provider = ElectricityMapsCarbonIntensityProvider(latlong)

    with pytest.raises(Exception):
        provider.get_current_carbon_intensity()


def test_get_lat_long_excpetion(mock_exception_ip):
    with pytest.raises(ConnectionError):
        get_ip_lat_long()
