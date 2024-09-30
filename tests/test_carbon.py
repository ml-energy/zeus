from __future__ import annotations

import json
import pytest
import requests

from unittest.mock import patch

from zeus.carbon import (
    ElectrictyMapsClient,
    get_ip_lat_long,
    CarbonIntensityNotFoundError,
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

    with pytest.raises(CarbonIntensityNotFoundError):
        provider.get_current_carbon_intensity()
