from dataclasses import dataclass
# from numpy import fft, abs
from numpy.random import randn
from scipy.fft import fft, fftfreq
from scipy.signal import windows, periodogram, correlate
from matplotlib import pyplot as plt
from utils import load_msft, Parameters, select_data_time
from analyses import fit_gp, gp_prediction_vs_martingale
from plot_tools import plot_data, plot_sliding_window_autocorr
import numpy as np


def run_initial_example():
    parameters = Parameters(num_data_points_gp_fit=50,
                            start_date="1989-06-01",
                            end_date="1989-12-31",
                            test_data_size=0.2,
                            rbf_length_scale=10.0,
                            rbf_output_scale=5.0,
                            tick_interval_x=50,
                            use_return=False,
                            sigma_price=0.01,
                            prediction_horizon=0)
    # Load dataset
    raw_data = load_msft(parameters)
    fit_gp(raw_data, parameters, prediction_horizon_mode="day", subsample_timeframe=True, prediction_mode="all")


parameters = Parameters(num_data_points_gp_fit=50,
                        start_date="1980-01-01",
                        end_date="2024-12-31",
                        test_data_size=0.01,
                        rbf_length_scale=1.0,
                        rbf_output_scale=5.0,
                        tick_interval_x=50,
                        use_return=True,
                        sigma_return=0.001,
                        prediction_horizon=10,
                        target_label="High")

raw_data = load_msft(parameters)
raw_data = select_data_time(raw_data, parameters.start_date, parameters.end_date)
# fit_gp(raw_data, parameters, prediction_horizon_mode="day", subsample_timeframe=True, prediction_mode="all")
# plot_data(raw_data, "Return", "Date", plot_format="r", title="Plot highest stock price",mode="Full")
plot_sliding_window_autocorr(raw_data, "Return", "dt", window_size=180)


pass
