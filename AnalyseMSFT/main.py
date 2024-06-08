from dataclasses import dataclass

from utils import load_msft
from analyses import fit_gp_time_period_train_test_split


# Parameters
@dataclass
class parameters:
    test_data_size = 0.5
    rbf_length_scale = 5.0
    rbf_output_scale = 20.0
    sigma_price = 0.2  # Make sigma_price a function of time?
    sigma_return = 1.0
    target_price = "High"
    use_return = True
    plot_shading_mode = "2-sigma"
    start_date = "1990-01-01"
    end_date = "1995-12-31"


# Load dataset
raw_data = load_msft()

fit_gp_time_period_train_test_split(raw_data, parameters)
pass
