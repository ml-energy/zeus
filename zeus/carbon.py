import time
import math
import json
import requests
import pandas as pd
from pprint import pprint
from dateutil.relativedelta import relativedelta
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
    window_size = int(window_size)
    if window_size < 5:
        window_size = 5
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

def get_forecast_query_time_range(estimate_ep_time_min=5):
    '''
    minimum query time range is 5 minutes
    '''
    if estimate_ep_time_min < 5:
        estimate_ep_time_min = 5
    gmt_now = datetime.now(timezone.utc)
    curr_gmt_str = time_to_str(gmt_now)
    start_time_obj = round_up_time(gmt_now)
    start_time_str = time_to_str(start_time_obj)
    end_time_obj = start_time_obj + timedelta(minutes=estimate_ep_time_min)
    end_time_obj = round_up_time(end_time_obj)
    end_time_str = time_to_str(end_time_obj)
    return curr_gmt_str, start_time_str, end_time_str

def get_history_avg(ep_time=30*60):
    '''
    minimum query time range is 1 minutes (60 seconds), other the API will return an error
    '''
    if ep_time < 60:
        ep_time = 60
    gmt_now = datetime.now(timezone.utc)
    curr_gmt_str = time_to_str(gmt_now)
    prev_time_obj = gmt_now - timedelta(seconds=ep_time)
    prev_time_str = time_to_str(prev_time_obj)

    headers = {'accept': 'application/json',}
    params = {
        'location': 'centralus',
        'startTime': prev_time_str,
        'endTime': curr_gmt_str,
        
    }
    response = requests.get('https://carbon-aware-api.azurewebsites.net/emissions/average-carbon-intensity', params=params, headers=headers)
    return response.json()

def compute_carbon_emissions(power_consumption, carbon_intensity):
    '''
    power_consumption unit: J
    carbon_intensity unit: g/kWh
    1 J = 2.77778e-7 kWh
    return carbon_emissions, unit: g
    '''
    return power_consumption * carbon_intensity * 2.77778e-7

if __name__ == "__main__":
    estimate_ep_time = 30
    # curr_gmt_str, start_time_str, end_time_str = get_forecast_query_time_range(estimate_ep_time)

    # print("\nCurrent GMT Time: {}".format(curr_gmt_str))
    # print("Query start time:", start_time_str)
    # print("Query end time:  ", end_time_str)
    
    # print()
    # forecast = get_forecast(start_time_str, end_time_str, estimate_ep_time)
    # pprint(forecast)

    data = []
    while True:
        curr_gmt_str, start_time_str, end_time_str = get_forecast_query_time_range(estimate_ep_time)
        forecast = get_forecast(start_time_str, end_time_str, estimate_ep_time)
        data.append(forecast[0]['forecastData'][0])
        pprint(forecast[0]['forecastData'][0])
        print()
        df = pd.DataFrame(data)
        df.to_csv("carbon.csv", index=False)

        print("sleeping...zzz")
        time.sleep(30*60)