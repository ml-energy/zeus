"""Carbon intensity providers used for carbon-aware optimizers."""

from __future__ import annotations

import abc
import requests
from typing import Literal

from zeus.exception import ZeusBaseError
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


class CarbonIntensityNotFoundError(ZeusBaseError):
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

        !!! Note
            In some locations, there is no recent carbon intensity data. `self.estimate` can be used to approximate the carbon intensity in such cases.
        """
        try:
            url = (
                f"https://api.electricitymap.org/v3/carbon-intensity/latest?lat={self.lat}&lon={self.long}"
                + f"&disableEstimations={not self.estimate}&emissionFactorType={self.emission_factor_type}"
            )
            resp = requests.get(url)

            return resp.json()["carbonIntensity"]
        except KeyError as e:
            # Raise exception when carbonIntensity does not exist in response
            raise CarbonIntensityNotFoundError(
                f"Recent carbon intensity measurement not found at `({self.lat}, {self.long})` "
                f"with estimate set to `{self.estimate}` and emission_factor_type set to `{self.emission_factor_type}`"
            ) from e
        except requests.exceptions.RequestException as e:
            logger.exception(
                "Failed to retrieve recent carbon intensnity measurement: %s", e
            )
            raise
