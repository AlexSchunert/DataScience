from dataclasses import dataclass

from utils import load_msft, Parameters
from analyses import fit_gp, gp_prediction_vs_martingale
from plot_tools import plot_data

# Parameters
parameters = Parameters(num_data_points_gp_fit=50,
                        start_date="1980-01-01",
                        end_date="1990-12-31",
                        test_data_size=0.2,
                        rbf_length_scale=5.0,
                        rbf_output_scale=10.0,
                        tick_interval_x=365)
# Load dataset
raw_data = load_msft(parameters)

# Process stuff
# plot_data(raw_data, "High", "Date", plot_format="r", title="Plot highest stock price")
fit_gp(raw_data, parameters, prediction_horizon_mode="day", subsample_timeframe=True, prediction_mode="all")
# gp_prediction_vs_martingale(raw_data, parameters, plot_iterations=True)

pass
