import numpy as np
from random import randint
import pandas as pd
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


def gp_prediction_vs_martingale(raw_data, parameters, plot_iterations=False):
    # In case the target quantity is the Return, compute it
    if parameters.use_return:
        raw_data = compute_return(raw_data, parameters.target_price)
        raw_data = raw_data.reset_index(drop=True)
        target_quantity_idx = "Return"
        result_label = "Return"
        sigma_target_quantity = parameters.sigma_return
    else:
        target_quantity_idx = parameters.target_price
        result_label = parameters.target_price
        sigma_target_quantity = parameters.sigma_price


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

        train_data = train_data[["Date", "dt", target_quantity_idx]]
        test_data = test_data[["Date", "dt", target_quantity_idx]]

        if plot_iterations:
            test_data = pd.concat([test_data, train_data]).sort_index()

        # Fit gp
        result = gp_process(test_data,
                            train_data,
                            target_quantity_idx,
                            result_label,
                            sigma_target_quantity,
                            parameters.rbf_length_scale,
                            parameters.rbf_output_scale)

        # calculate prediction and store
        if parameters.use_return:
            prediction_error[i] = result[result_label].values[-1] - \
                                  test_data[target_quantity_idx].values[-1]
            martingale_error[i] = -test_data[target_quantity_idx].values[-1]
        else:
            prediction_error[i] = result[result_label].values[-1] - \
                                  test_data[target_quantity_idx].values[-1]
            martingale_error[i] = train_data[target_quantity_idx].values[-1] - \
                                  test_data[target_quantity_idx].values[-1]

        if plot_iterations:
            # plot result
            plot_prediction_result(train_data,
                                   test_data,
                                   result,
                                   target_quantity_idx,
                                   result_idx=result_label,
                                   plot_shading_mode=parameters.plot_shading_mode)


    plot_prediction_error_statistic(prediction_error,
                                    reference_error=martingale_error,
                                    num_bins=parameters.histogram_num_bins)
