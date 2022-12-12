"""Carbon Emissions API."""

import time
import math
import json
import requests
import pandas as pd
from pprint import pprint
from dateutil.relativedelta import relativedelta
from datetime import timedelta, datetime, timezone


def round_up_time(time_obj: datetime, query_limit_min: int = 5) -> datetime:
    """
    Given a datetime object, round up the time to the nearest 5 minutes.

    Args:
    time_obj: datetime object
    query_limit_min: the minimum query time range is 5 minutes
    """
    rounded_minute = query_limit_min * math.ceil(
        (time_obj.minute / query_limit_min) + 1e-12
    )
    time_obj += timedelta(minutes=rounded_minute - time_obj.minute)
    return time_obj


def time_to_str(time_obj) -> str:
    """
    Given a datetime object, return a string in the format of "YYYY-MM-DDTHH:MM:00Z" matches the format of API query.

    Args:
    time_obj: datetime object
    """
    # current GMT Time == current utc time
    # gmt = datetime.now(timezone.utc)
    return "{}-{}-{}T{:02}:{:02}:00Z".format(
        time_obj.year, time_obj.month, time_obj.day, time_obj.hour, time_obj.minute
    )


def get_forecast(start_time: str, end_time: str, window_size: int) -> dict:
    """
    Given a start time, end time, and window size, return a dictionary of forecasted carbon emission.

    Args:
    start_time: start time of the query in the format of "YYYY-MM-DDTHH:MM:00Z"
    end_time: end time of the query in the format of "YYYY-MM-DDTHH:MM:00Z"
    window_size: window size of the query in minutes
    """
    window_size = int(window_size)
    if window_size < 5:
        window_size = 5
    headers = {
        "accept": "application/json",
    }
    params = {
        # Azure regions can be found here: https://learn.microsoft.com/en-us/azure/availability-zones/az-overview
        "location": "centralus",  # Can be obtained through from Azure API
        "dataStartAt": start_time,
        "dataEndAt": end_time,
        "windowSize": window_size,
    }
    response = requests.get(
        "https://carbon-aware-api.azurewebsites.net/emissions/forecasts/current",
        params=params,
        headers=headers,
    )
    return response.json()


def get_forecast_query_time_range(estimate_ep_time_min: int = 5) -> tuple:
    """
    Minimum query time range is 5 minutes.

    Args:
    estimate_ep_time_min: estimated episode time in minutes
    """
    if estimate_ep_time_min < 5:
        estimate_ep_time_min = 5
    gmt_now = datetime.now(timezone.utc)
    curr_gmt_str = time_to_str(gmt_now)
    start_time_obj = round_up_time(gmt_now)
    start_time_str = time_to_str(start_time_obj)
    end_time_obj = start_time_obj + timedelta(minutes=estimate_ep_time_min)
    end_time_obj = round_up_time(end_time_obj)
    end_time_str = time_to_str(end_time_obj)
    return (curr_gmt_str, start_time_str, end_time_str)


def get_history_avg(ep_time: int = 30 * 60) -> dict:
    """
    Minimum query time range is 1 minutes (60 seconds), other the API will return an error.

    Args:
    ep_time: estimated episode time in seconds
    """
    if ep_time < 60:
        ep_time = 60
    gmt_now = datetime.now(timezone.utc)
    curr_gmt_str = time_to_str(gmt_now)
    prev_time_obj = gmt_now - timedelta(seconds=ep_time)
    prev_time_str = time_to_str(prev_time_obj)

    headers = {
        "accept": "application/json",
    }
    params = {
        "location": "centralus",
        "startTime": prev_time_str,
        "endTime": curr_gmt_str,
    }
    response = requests.get(
        "https://carbon-aware-api.azurewebsites.net/emissions/average-carbon-intensity",
        params=params,
        headers=headers,
    )
    return response.json()


def compute_carbon_emissions(
    power_consumption: float, carbon_intensity: float
) -> float:
    """
    Compute carbon emissions in g given power consumption in Joules and carbon intensity in g/kWh.

    Args:
    power_consumption: power consumption in Joules
    carbon_intensity: carbon intensity in g/kWh
    """
    return power_consumption * carbon_intensity * 2.77778e-7


# testing
if __name__ == "__main__":
    estimate_ep_time = 30

    data = []
    while True:
        curr_gmt_str, start_time_str, end_time_str = get_forecast_query_time_range(
            estimate_ep_time
        )
        forecast = get_forecast(start_time_str, end_time_str, estimate_ep_time)
        data.append(forecast[0]["forecastData"][0])
        pprint(forecast[0]["forecastData"][0])
        print()
        df = pd.DataFrame(data)
        df.to_csv("carbon.csv", index=False)

        print("sleeping...zzz")
        time.sleep(30 * 60)
