"""Function for getting latitude and longitude."""

import requests
import logging

logger = logging.getLogger(__name__)


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
        logger.exception("Failed to retrieve current latitude and longitude of IP: %s", e)
        raise
