from gaussian_process import condition_gpr, predict_gpr
from utils import load_msft, select_data_time, train_test_split, compute_kernel_matrices, construct_prediction_result
from plot_tools import plot_prediction_result

# Parameters
test_data_size = 0.5
rbf_length_scale = 10.0
rbf_output_scale = 20.0
sigma_price = 0.2  # Make sigma_price a function of time?
target_price = "High"
plot_shading_mode = "2-sigma"
start_date = "2000-01-01"
end_date = "2005-12-31"

# Load dataset
raw_data = load_msft()
# Split into train- and test-data
data = select_data_time(raw_data, start_date, end_date)
# Remove unnecessary colums from data
data = data[["Date", "dt", target_price]]
# split into training and test data
train_data, test_data = train_test_split(data, test_data_size)
# Compute kernel matrices
k_xx, k_zx, k_zz = compute_kernel_matrices(data, train_data, rbf_length_scale, rbf_output_scale, "rbf")
# condition on data
alpha, predictive_cov = condition_gpr(train_data, k_xx, target_price, sigma_price)
# predict for all data
price_predicted, std_prediction = predict_gpr(alpha, k_zx, k_zz, predictive_cov)
# predict for all data
result = construct_prediction_result(data, price_predicted, std_prediction)
# plot result
plot_prediction_result(train_data, test_data, result, target_price, plot_shading_mode)

pass
