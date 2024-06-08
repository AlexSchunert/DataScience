import numpy as np
from random import randint
import pandas as pd
from datetime import datetime, timedelta

from utils import train_test_split, compute_kernel_matrices, construct_prediction_result, compute_return, \
    select_data_time
from plot_tools import plot_prediction_result, plot_prediction_error_statistic
from gaussian_process import gp_process


def fit_gp(data,
           parameters,
           subsample_timeframe=False,
           start_date=None,
           end_date=None,
           prediction_horizon=None,
           prediction_horizon_mode="days",
           predict_only=False,
           plot_results=True):

    # Check inputs
    if start_date is None:
        start_date = parameters.start_date
    if end_date is None:
        end_date = parameters.end_date
    if prediction_horizon is None:
        prediction_horizon = parameters.prediction_horizon

    # Set name of result
    result_label = parameters.target_label

    # Get rid of unnecessary columns
    data = data[["Date", "dt", parameters.target_label]]

    # Get selected timeframe
    data_timeframe = select_data_time(data, start_date, end_date)

    # split into training and test data if subsampling is enabled
    if subsample_timeframe:
        data_timeframe_train, data_timeframe_test = train_test_split(data_timeframe, parameters.test_data_size)
    else:
        data_timeframe_train = data_timeframe
        data_timeframe_test = pd.DataFrame(columns=data_timeframe_train.columns)

    # Construct prediction data
    if predict_only:
        data_timeframe_test = pd.DataFrame(columns=data_timeframe_train.columns)

    if prediction_horizon_mode == "index":
        idx_date_larger = np.where(data["Date"] > end_date)
        if idx_date_larger is None:
            data_timeframe_predict = pd.DataFrame(columns=data_timeframe_train.columns)
        elif np.where((data["Date"] >= end_date))[0].shape[0] == 0:
            data_timeframe_predict = pd.DataFrame(columns=data_timeframe_train.columns)
        else:
            idx_start_prediction = np.min(idx_date_larger[0])
            idx_end_prediction = np.min([idx_start_prediction + prediction_horizon - 1, data.shape[0]])
            data_timeframe_predict = data.loc[idx_start_prediction:idx_end_prediction]

    else:
        # Select and concatenate with data_timeframe_test
        start_date_prediction = (datetime.strptime(parameters.end_date, "%Y-%m-%d") + timedelta(1)).strftime("%Y-%m-%d")
        end_date_prediction = (
                    datetime.strptime(parameters.end_date, "%Y-%m-%d") + timedelta(prediction_horizon)).strftime(
            "%Y-%m-%d")
        data_timeframe_predict = select_data_time(data, start_date_prediction, end_date_prediction)

    if data_timeframe_test.empty == False or data_timeframe_predict.empty == False:
        data_timeframe_test = pd.concat([df for df in [data_timeframe_test, data_timeframe_predict] if not df.empty]).\
            sort_index()
    else:
        return

    result = gp_process(data_timeframe_test,
                        data_timeframe_train,
                        parameters.target_label,
                        result_label,
                        parameters.sigma_used,
                        parameters.rbf_length_scale,
                        parameters.rbf_output_scale)

    if plot_results:
        plot_prediction_result(data_timeframe_train,
                               data_timeframe_test,
                               result,
                               parameters.target_label,
                               result_idx=result_label,
                               plot_shading_mode=parameters.plot_shading_mode)

    return result


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
