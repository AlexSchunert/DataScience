from utils import load_msft, select_data_time, compute_return
from analyses import fit_gp_standard

# Parameters
test_data_size = 0.5
rbf_length_scale = 1.0
rbf_output_scale = 10.0
sigma_price = 0.2  # Make sigma_price a function of time?
sigma_return = 5.0
target_price = "High"
use_return = True
plot_shading_mode = "2-sigma"
start_date = "1990-01-01"
end_date = "1990-12-31"

# Load dataset
raw_data = load_msft()

# In case the target quantity is the Return, compute it
if use_return:
    raw_data = compute_return(raw_data, target_price)

# Split into train- and test-data
data = select_data_time(raw_data, start_date, end_date)

if not use_return:
    # Remove unnecessary colums from data
    data = data[["Date", "dt", target_price]]
    # Fit gp
    fit_gp_standard(data,
                    target_price,
                    test_data_size,
                    rbf_length_scale,
                    rbf_output_scale,
                    sigma_price,
                    plot_shading_mode)

else:
    # Remove unnecessary colums from data
    data = data[["Date", "dt", "Return"]]
    # Fit gp
    fit_gp_standard(data,
                    "Return",
                    test_data_size,
                    rbf_length_scale,
                    rbf_output_scale,
                    sigma_return,
                    plot_shading_mode)

pass
