"""Abstract Carbon Intensity Provider Class."""
import abc


class CarbonIntensityProvider(abc.ABC):
    """Abstract class for implementing ways to fetch carbon intensity."""

    def __init__(self, location: tuple[float, float]) -> None:
        """Initializes carbon intensity provider location to the latitude and longitude of the input `location`.

        Location is a tuple of floats where latitude is the first float and longitude is the second float.
        """
        self.lat = location[0]
        self.long = location[1]

    @abc.abstractmethod
    def get_current_carbon_intensity(self) -> float:
        """Abstract method for fetching the current carbon intensity of the set location of the class."""
        pass
