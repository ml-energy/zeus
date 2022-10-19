import time
import math
import requests
from pprint import pprint
from datetime import timedelta, datetime, timezone


# headers = {
#     'accept': 'application/json',
# }

# params = {
#     'location': 'eastus',
#     'dataStartAt': '2022-10-17T04:55:00Z',
#     'dataEndAt': '2022-10-17T05:30:00Z',
#     'windowSize': '30',
# }

# response = requests.get('https://carbon-aware-api.azurewebsites.net/emissions/forecasts/current', 
#                         params=params, headers=headers)

# def main():
#     print(response.json())

def round_up_time(time_obj, query_limit_min=5):
    rounded_minute = query_limit_min * math.ceil((time_obj.minute/query_limit_min) + 1e-12)
    time_obj += timedelta(minutes=rounded_minute - time_obj.minute)
    return time_obj

def time_to_str(time_obj):
    # current GMT Time == current utc time
    # gmt = datetime.now(timezone.utc)
    return "{}-{}-{}T{:02}:{:02}:00Z".format(time_obj.year, time_obj.month, time_obj.day, time_obj.hour, time_obj.minute)

def get_forecast(start_time, end_time, window_size):
    headers = {'accept': 'application/json',}
    params = {
        # Azure regions can be found here: https://learn.microsoft.com/en-us/azure/availability-zones/az-overview
        'location': 'centralus', # Can be obtained through from Azure API
        'dataStartAt': start_time,
        'dataEndAt': end_time,
        'windowSize': window_size,
    }
    response = requests.get('https://carbon-aware-api.azurewebsites.net/emissions/forecasts/current', 
                        params=params, headers=headers)
    return response.json()


if __name__ == "__main__":
    estimate_ep_time = 5
    gmt_now = datetime.now(timezone.utc)
    curr_gmt_str = time_to_str(gmt_now)

    start_time_obj = round_up_time(gmt_now)
    start_time_str = time_to_str(start_time_obj)
    end_time_obj = start_time_obj + timedelta(minutes=estimate_ep_time)

    end_time_obj = round_up_time(end_time_obj)
    end_time_str = time_to_str(end_time_obj)

    print("Current GMT Time: {}".format(curr_gmt_str))
    print("Query start time:", start_time_str)
    print("Query end time:  ", end_time_str)
    
    print()
    forecast = get_forecast(start_time_str, end_time_str, estimate_ep_time)
    pprint(forecast)

