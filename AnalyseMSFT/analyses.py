from utils import train_test_split, compute_kernel_matrices, construct_prediction_result, compute_return, \
    select_data_time
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


def fit_gp_time_period_train_test_split(raw_data, parameters):
    # In case the target quantity is the Return, compute it
    if parameters.use_return:
        raw_data = compute_return(raw_data, parameters.target_price)

    # Split into train- and test-data
    data = select_data_time(raw_data, parameters.start_date, parameters.end_date)

    if not parameters.use_return:
        # Remove unnecessary colums from data
        data = data[["Date", "dt", parameters.target_price]]
        # Fit gp
        fit_gp_standard(data,
                        parameters.target_price,
                        parameters.test_data_size,
                        parameters.rbf_length_scale,
                        parameters.rbf_output_scale,
                        parameters.sigma_price,
                        parameters.plot_shading_mode)

    else:
        # Remove unnecessary colums from data
        data = data[["Date", "dt", "Return"]]
        # Fit gp
        fit_gp_standard(data,
                        "Return",
                        parameters.test_data_size,
                        parameters.rbf_length_scale,
                        parameters.rbf_output_scale,
                        parameters.sigma_return,
                        parameters.plot_shading_mode)

# def gp_prediction_vs_
