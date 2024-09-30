"""Carbon Intensity Provider using ElectrictyMaps API."""
from zeus.carbon.carbon_intensity_provider import CarbonIntensityProvider
import requests


class ElectricityMapsCarbonIntensityProvider(CarbonIntensityProvider):
    """Carbon Intensity Provider with ElectricityMaps API."""

    def get_current_carbon_intensity(
        self, estimate: bool = False, emission_factor_type: str = "direct"
    ) -> float:
        """Fetches current carbon intensity of the location of the class.

        Args:
            estimate: bool to toggle whether carbon intensity is estimated or not
            emission_factor_type: emission factor to be measured (`direct` or `lifestyle`)

        !!! Note
            In some locations, there is no recent carbon intensity data. `estimate` can be used to approximate the carbon intensity in such cases.
        """
        try:
            url = (
                f"https://api.electricitymap.org/v3/carbon-intensity/latest?lat={self.lat}&lon={self.long}"
                + f"&disableEstimations={not estimate}&emissionFactorType={emission_factor_type}"
            )
            resp = requests.get(url)
            return resp.json()["carbonIntensity"]
        except Exception as e:
            print(f"Failed to retrieve live carbon intensity data: {e}")
            raise (e)
