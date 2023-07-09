"""Defines the carbon_monitor class."""

import numpy as np
import pandas as pd

# from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import (
#     GradientBoostingRegressor,
#     AdaBoostRegressor,
#     RandomForestRegressor,
# )


class CarbonMonitor:
    """Walk through the carbon trace and predict the carbon intensity."""

    def __init__(self, interval_min: int = 30, start_step: int = 48) -> None:
        """Initialize the carbon_monitor."""
        self.df = pd.read_csv("/workspace/zeus/trace/carbon_hist.csv")
        self.total_steps_day = (24 * 60) / interval_min
        self.step_counter = start_step

        # data processing
        cycle = np.arange(1, 49)
        cycles5 = np.concatenate([cycle] * 5)
        self.df["time_enc"] = cycles5[: len(self.df)]
        self.df["sin_time"] = np.sin(
            2 * np.pi * self.df.time_enc / self.total_steps_day
        )
        self.df["cos_time"] = np.cos(
            2 * np.pi * self.df.time_enc / self.total_steps_day
        )
        self.df["prev"] = get_prev(self.df["hist"].values, 1)
        x_inputs = self.df[["sin_time", "cos_time", "prev"]]
        y_outputs = self.df[["hist"]]
        x_train, y_train = (
            x_inputs.values[start_step - 48 : start_step],
            y_outputs.values.flatten()[start_step - 48 : start_step],
        )
        # X_test, y_test = x_inputs.values[start_step:], y_outputs.values.flatten()[start_step:]

        # train
        model = SVR(kernel="rbf", C=100, gamma=0.0001)
        model.fit(x_train, y_train)
        self.model = model

    def update_steps(self, steps) -> None:
        """Walk #step along the trace."""
        self.step_counter += steps
        if self.step_counter > len(self.df):
            self.step_counter = 1
            print("You have reached the end of the trace!!!")

    def get_hist(self) -> float:
        """Return the carbon intensity at the current step."""
        return self.df["hist"][self.step_counter - 1]

    def get_forecast(self) -> float:
        """Return the carbon intensity at the next step."""
        curr_time_step = self.step_counter
        input_ = [
            self.df.iloc[curr_time_step]["sin_time"],
            self.df.iloc[curr_time_step]["cos_time"],
            self.df.iloc[curr_time_step]["prev"],
        ]
        return self.model.predict([input_])[0]


def carbon_cost(
    power,
    tput,
    power_limits,
    next_carbon_intensity,
    max_carbon=750,
    max_pl=300,
    eta_knob=0.24,
    world_size=1,
    return_feature=False,
):
    """Compute the cost of different power limits."""
    time_nor_coeff = max_pl * max_carbon
    cost_map = {
        pl: (
            eta_knob * power[pl] * next_carbon_intensity
            + (1 - eta_knob) * time_nor_coeff * world_size
        )
        / tput[pl]
        for pl in power_limits
    }
    optimal_pl = min(cost_map.keys(), key=cost_map.get)

    if return_feature:
        cost_feature_map = {
            pl: [eta_knob * power[pl] * next_carbon_intensity] for pl in power_limits
        }
        const_part = (1 - eta_knob) * time_nor_coeff * world_size
        return optimal_pl, cost_map, cost_feature_map, const_part
    return optimal_pl, cost_map


def get_prev(hist_list, shift, up=True):
    """Get the previous value of the carbon intensity."""
    if up:
        tmp_list = hist_list[:-shift]
        back = hist_list[-shift:]
        prev_fea = np.insert(tmp_list, 0, back)
    else:
        tmp_list = hist_list[shift:]
        front = hist_list[:shift]
        prev_fea = np.insert(tmp_list, 0, front)
    return prev_fea


def recursive_forecasting(model, x_test):
    """Recursive forecasting."""
    steps = len(x_test) - 1
    if len(x_test[0]) > 3:
        forecast = [x_test[0][2:]]  # first 2 are time, the rest are carbon
    else:
        forecast = [x_test[0][-1]]
    for i in range(steps):
        next_time_sin, next_time_cos = x_test[i][0], x_test[i][1]
        last_val = forecast[-1]
        model_input = [next_time_sin, next_time_cos]
        model_input = np.append(model_input, last_val)
        pred_one = model.predict([model_input])[0]
        # print(model_input[2:])
        # print(pred_one)
        # print()
        if len(x_test[0]) > 3:
            forecast.append(np.append(last_val[1:], pred_one))
        else:
            forecast.append(pred_one)
    if len(x_test[0]) > 3:
        forecast = np.array(forecast)
        return forecast[:, 0]
    return forecast


def recur_hist_forecasting(model, x_test, skip_step=1):
    """Recursive forecasting with history."""
    if skip_step == -1:
        return recursive_forecasting(model, x_test)
    steps = len(x_test) - 1
    if len(x_test[0]) > 3:
        forecast = [x_test[0][2:]]
    else:
        forecast = [x_test[0][-1]]

    for i in range(steps):
        next_time_sin, next_time_cos = x_test[i][0], x_test[i][1]
        last_val = forecast[-1]

        if skip_step == "random" and np.random.randint(0, 2) and i != 0:
            if len(x_test[0]) > 3:
                last_val = x_test[i][2:]
            else:
                last_val = x_test[i][-1]
        elif isinstance(skip_step) == int and i % skip_step == 0 and i != 0:
            if len(x_test[0]) > 3:
                last_val = x_test[i][2:]
            else:
                last_val = x_test[i][-1]

        model_input = [next_time_sin, next_time_cos]
        model_input = np.append(model_input, last_val)
        pred_one = model.predict([model_input])[0]
        if len(x_test[0]) > 3:
            forecast.append(np.append(last_val[1:], pred_one))
        else:
            forecast.append(pred_one)

    if len(x_test[0]) > 3:
        forecast = np.array(forecast)
        return forecast[:, 0]
    return forecast


def compute_carbon_emissions(
    energy_consumption: float, carbon_intensity: float
) -> float:
    """
    Compute carbon emissions in g given energy consumption in Joules and carbon intensity in g/kWh.

    Args:
    energy_consumption: power consumption in Joules
    carbon_intensity: carbon intensity in g/kWh
    """
    return energy_consumption * carbon_intensity * 2.77778e-7
