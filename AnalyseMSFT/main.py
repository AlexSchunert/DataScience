from dataclasses import dataclass

from utils import load_msft
from analyses import fit_gp, gp_prediction_vs_martingale


# Parameters
@dataclass
class parameters:
    test_data_size: float = 0.5
    rbf_length_scale: float = 5.0#20.0
    rbf_output_scale: float = 20.0
    sigma_price: float = 0.1  # Make sigma_price a function of time?
    sigma_return: float = 1.0
    sigma_used: float = 0.0
    target_label: str = "High"
    use_return: bool = True
    plot_shading_mode: str = "2-sigma"
    start_date: str = "2000-01-01"
    end_date: str = "2006-12-31"
    prediction_horizon: int = 10
    num_iter_error_stat: int = 1000
    num_data_points_gp_fit: int = 10
    histogram_num_bins: int = 100


# Load dataset
raw_data = load_msft(parameters)

fit_gp(raw_data, parameters, prediction_horizon_mode="day", subsample_timeframe=True, predict_only=True)
# gp_prediction_vs_martingale(raw_data, parameters, plot_iterations=False)

pass
