from utils import train_test_split, compute_kernel_matrices, construct_prediction_result
from plot_tools import plot_prediction_result
from gaussian_process import condition_gpr, predict_gpr

def fit_gp_standard(data,
                    target_price,
                    test_data_size,
                    rbf_length_scale,
                    rbf_output_scale,
                    sigma_price,
                    plot_shading_mode):

    # split into training and test data
    train_data, test_data = train_test_split(data, test_data_size)
    # Compute kernel matrices
    k_xx, k_zx, k_zz = compute_kernel_matrices(data, train_data, rbf_length_scale, rbf_output_scale, "rbf")
    # condition on data
    alpha, predictive_cov = condition_gpr(train_data, k_xx, target_price, sigma_price)
    # predict for all data
    price_predicted, std_prediction = predict_gpr(alpha, k_zx, k_zz, predictive_cov)
    # create result
    result = construct_prediction_result(data, price_predicted, std_prediction)
    # plot result
    plot_prediction_result(train_data, test_data, result, target_price, plot_shading_mode)
