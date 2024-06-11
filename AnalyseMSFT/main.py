from dataclasses import dataclass

from utils import load_msft, Parameters
from analyses import fit_gp, gp_prediction_vs_martingale
from plot_tools import plot_data

# Parameters
parameters = Parameters()
# Load dataset
raw_data = load_msft(parameters)
# plot_data(raw_data, "High", "Date", plot_format="r", title="Plot highest stock price")
# fit_gp(raw_data, parameters, prediction_horizon_mode="day", subsample_timeframe=True, prediction_mode="predict_only")
gp_prediction_vs_martingale(raw_data, parameters, plot_iterations=True)

pass
