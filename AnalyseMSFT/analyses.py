import numpy as np
from random import randint

from utils import train_test_split, compute_kernel_matrices, construct_prediction_result, compute_return, \
    select_data_time, select_data_idx
from plot_tools import plot_prediction_result, plot_prediction_error_statistic
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


def gp_predict(test_data, train_data, target_quantity, sigma_measurement, rbf_length_scale, rbf_output_scale):
    # Fit gp
    k_xx, k_zx, k_zz = compute_kernel_matrices(test_data,
                                               train_data,
                                               rbf_length_scale,
                                               rbf_output_scale,
                                               "rbf")
    # condition on data
    alpha, predictive_cov = condition_gpr(train_data, k_xx, target_quantity, sigma_measurement)
    # predict for all data
    price_predicted, std_prediction = predict_gpr(alpha, k_zx, k_zz, predictive_cov)
    # create result
    result = construct_prediction_result(test_data, price_predicted, std_prediction)

    return result


def gp_prediction_vs_martingale(raw_data, parameters):
    # In case the target quantity is the Return, compute it
    if parameters.use_return:
        raw_data = compute_return(raw_data, parameters.target_price)
        raw_data = raw_data.reset_index(drop=True)

    # create output arrays
    prediction_error = np.zeros(parameters.num_iter_error_stat, )
    martingale_error = np.zeros(parameters.num_iter_error_stat)

    # interval is selected based on its endpoint => max and min allowable end indices
    max_idx_end = raw_data.shape[0] - 2
    min_idx_end = parameters.num_data_points_gp_fit - 1

    # 1) Loop over number randomly selected chunks with fixed length
    # 2) In each iteration: Predict next day => Calculate prediction error+Calculate martingale prediction error => save

    for i in range(parameters.num_iter_error_stat):

        # Select data from randomly selected timeframe
        idx_end = randint(min_idx_end, max_idx_end)
        train_data = raw_data.loc[idx_end - parameters.num_data_points_gp_fit + 1:idx_end]
        test_data = raw_data.loc[[idx_end + 1]]
        if not parameters.use_return:
            # Remove unnecessary colums from data
            train_data = train_data[["Date", "dt", parameters.target_price]]
            test_data = test_data[["Date", "dt", parameters.target_price]]
            # Fit gp
            result = gp_predict(test_data,
                                train_data,
                                parameters.target_price,
                                parameters.sigma_price,
                                parameters.rbf_length_scale,
                                parameters.rbf_output_scale)

            # calculate prediction and store
            prediction_error[i] = result["price_prediction"].values[0] - test_data[parameters.target_price].values[0]
            martingale_error[i] = train_data[parameters.target_price].values[-1] - \
                                  test_data[parameters.target_price].values[0]

        else:
            # Remove unnecessary colums from data
            train_data = train_data[["Date", "dt", "Return"]]
            test_data = test_data[["Date", "dt", "Return"]]

            result = gp_predict(test_data,
                                train_data,
                                "Return",
                                parameters.sigma_return,
                                parameters.rbf_length_scale,
                                parameters.rbf_output_scale)

            # calculate prediction and store
            prediction_error[i] = result["price_prediction"].values[0] - test_data["Return"].values[0]
            martingale_error[i] = -test_data["Return"].values[0]

    plot_prediction_error_statistic(prediction_error, reference_error=martingale_error)
