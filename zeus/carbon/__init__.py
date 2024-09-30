"""Carbon intensity providers used for carbon-aware optimizers."""

from zeus.carbon.electricity_maps_carbon_intensity_provider import (
    ElectricityMapsCarbonIntensityProvider,
)

import requests


def get_ip_lat_long() -> tuple[float, float]:
    """Retrieve the latitude and longitude of the current IP position."""
    try:
        ip_url = "http://ipinfo.io/json"
        resp = requests.get(ip_url)
        loc = resp.json()["loc"]
        lat, long = map(float, loc.split(","))
        print(f"Retrieve latitude and longitude: {lat}, {long}")
        return lat, long
    except Exception as e:
        print(f"Failed to Retrieve Current IP's Latitude and Longitude: {e}")
        raise (e)
