import numpy as np
from random import randint

from utils import train_test_split, compute_kernel_matrices, construct_prediction_result, compute_return, \
    select_data_time
from plot_tools import plot_prediction_result, plot_prediction_error_statistic
from gaussian_process import gp_process


def fit_gp_time_period_train_test_split(raw_data, parameters):
    # In case the target quantity is the Return, compute it
    if parameters.use_return:
        raw_data = compute_return(raw_data, parameters.target_price)

    # Get selected timeframe
    data = select_data_time(raw_data, parameters.start_date, parameters.end_date)
    # split into training and test data
    train_data, test_data = train_test_split(data, parameters.test_data_size)

    if not parameters.use_return:
        # Remove unnecessary colums from data
        data = data[["Date", "dt", parameters.target_price]]
        result_label = parameters.target_price
        # Fit gp
        result = gp_process(data,
                            train_data,
                            parameters.target_price,
                            result_label,
                            parameters.sigma_price,
                            parameters.rbf_length_scale,
                            parameters.rbf_output_scale)

        # plot result
        plot_prediction_result(train_data,
                               test_data,
                               result,
                               parameters.target_price,
                               result_idx=result_label,
                               plot_shading_mode=parameters.plot_shading_mode)

    else:
        # Remove unnecessary colums from data
        data = data[["Date", "dt", "Return"]]
        result_label = "Return"
        # Fit gp
        result = gp_process(data,
                            train_data,
                            "Return",
                            result_label,
                            parameters.sigma_price,
                            parameters.rbf_length_scale,
                            parameters.rbf_output_scale)

        # plot result
        plot_prediction_result(train_data,
                               test_data,
                               result,
                               "Return",
                               result_idx=result_label,
                               plot_shading_mode=parameters.plot_shading_mode)


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
            result_label = parameters.target_price
            # Fit gp
            result = gp_process(test_data,
                                train_data,
                                parameters.target_price,
                                result_label,
                                parameters.sigma_price,
                                parameters.rbf_length_scale,
                                parameters.rbf_output_scale)

            # calculate prediction and store
            prediction_error[i] = result[result_label].values[0] - \
                                  test_data[parameters.target_price].values[0]
            martingale_error[i] = train_data[parameters.target_price].values[-1] - \
                                  test_data[parameters.target_price].values[0]

        else:
            # Remove unnecessary colums from data
            train_data = train_data[["Date", "dt", "Return"]]
            test_data = test_data[["Date", "dt", "Return"]]
            result_label = "Return"

            result = gp_process(test_data,
                                train_data,
                                "Return",
                                result_label,
                                parameters.sigma_return,
                                parameters.rbf_length_scale,
                                parameters.rbf_output_scale)

            # calculate prediction and store
            prediction_error[i] = result[result_label].values[0] - test_data["Return"].values[0]
            martingale_error[i] = -test_data["Return"].values[0]

    plot_prediction_error_statistic(prediction_error,
                                    reference_error=martingale_error,
                                    num_bins=parameters.histogram_num_bins)
