import numpy as np
from random import randint
import pandas as pd
from datetime import datetime, timedelta

from utils import train_test_split, construct_prediction_result, compute_return, \
    select_data_time, Parameters
from plot_tools import plot_prediction_result, plot_prediction_error_statistic, plot_gp_analysis
from gp_main import gp_process
from acf_tools import compute_gp_kernel_posterior


def fit_gp(data,
           parameters,
           subsample_timeframe=False,
           start_date=None,
           end_date=None,
           prediction_horizon=None,
           prediction_horizon_mode="days",
           prediction_mode="all",
           plot_results=True):
    """
    General interface function to fit gp to timeseries

    :param data: Dataset used to fit GP. Must contain columns labeled "Date", dt, and parameters.target_label
    :type data: pd.DataFrame
    :param parameters: Contains general settings. May be overwritten by function params
    :type parameters: Parameters
    :param subsample_timeframe: If true, the data in the timeframe [start_date,end_date] is sub-sampled.
                                The portion of data used as test-data is given by parameters.test_data_size.
    :type subsample_timeframe: bool
    :param start_date: Start of considered timeframe. If there is a data-point with exactly this date, it is included.
                       If None, the corresponding value from parameters is used.
    :type start_date: str
    :param end_date: End of considered timeframe. If there is a data-point with exactly this date, it is included
                     If None, the corresponding value from parameters is used.
    :type end_date: str
    :param prediction_horizon: Determines how far into the future predictions are made. Depending on
                               prediction_horizon_mode, the value either specifies days in the future or
                               number of (next) future data points. Note that data points must exist in data.
    :type prediction_horizon: int
    :param prediction_horizon_mode: Either "days" => prediction of prediction_horizon days into the future or
                                    "index" => next prediction_horizon points
    :type prediction_horizon_mode:  str
    :param prediction_mode: Either "all" => prediction is done for data from timeframe and for prediction points or
                            "predict_only" => prediction is done only for future points
    :type prediction_mode: str
    :param plot_results: If true, plot results
    :type plot_results: bool

    :return: DataFrame containing the columns: "Date", "dt", parameters.result_label, and "std". If prediction set
             is empty, None is returned.
    :rtype: pd.DataFrame
    """

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
    if prediction_mode == "all":
        data_timeframe_test = data_timeframe
    elif prediction_mode == "predict_only":
        data_timeframe_test = pd.DataFrame(columns=data_timeframe_train.columns)
    else:
        data_timeframe_test = data_timeframe

    if prediction_horizon == 0:
        data_timeframe_predict = pd.DataFrame(columns=data_timeframe_train.columns)
    elif prediction_horizon_mode == "index":
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

    if data_timeframe_test.empty is False or data_timeframe_predict.empty is False:
        data_timeframe_test = pd.concat([df for df in [data_timeframe_test, data_timeframe_predict] if not df.empty]). \
            sort_index()
    else:
        return

    if parameters.kernel_fct == "rbf":
        result, _ = gp_process(data_timeframe_test,
                               data_timeframe_train,
                               parameters.target_label,
                               result_label,
                               parameters.sigma_used,
                               rbf_length_scale=parameters.rbf_length_scale,
                               rbf_output_scale=parameters.rbf_output_scale,
                               kernel_fct=parameters.kernel_fct)
    elif parameters.kernel_fct == "gp_kernel":

        gp_posterior, sigma_measurement = compute_gp_kernel_posterior(data_timeframe_train, parameters.target_label)
        result, _ = gp_process(data_timeframe_test,
                               data_timeframe_train,
                               parameters.target_label,
                               result_label,
                               sigma_measurement,
                               gp_posterior=gp_posterior,
                               kernel_fct=parameters.kernel_fct)
    else:
        print("Unknown kernel_fct. Abort")
        return

    if plot_results:
        plot_gp_analysis(data_timeframe_train,
                         data_timeframe_test,
                         result,
                         parameters.target_label,
                         result_idx=result_label,
                         plot_shading_mode=parameters.plot_shading_mode,
                         tick_interval_x=parameters.tick_interval_x,
                         plot_line_tr_data=parameters.plot_line_tr_data,
                         plot_line_test_data=parameters.plot_line_test_data)

    return result


def gp_prediction_vs_martingale(raw_data, parameters, plot_iterations=False):
    """
    This function implements an experiment evaluating the usefulness of the implemented gaussian process regression
    for the prediction of stock prices or 1-day-returns, respectively. For that the algorithm samples random timeframes
    of fixed length from the data and predicts the stock price (or return) for the next day. The prediction error is
    then calculated using the known stock price (or return) from the data. For comparison, the gp prediction is compared
    to a constant stock price (or zero return), which corresponds to assuming a martingale time series for the stock
    price. The above procedure is repeated parameters.num_iter_error_stat time to build a statistic which is ultimately
    plotted in a histogram plot. Note that the length of the considered timeframe can be set using
    parameters.num_data_points_gp_fit.

    :param raw_data: Dataset used for analysis. Must contain columns labeled "Date", dt, and parameters.target_label
    :type raw_data: pd.DataFrame
    :param parameters: Contains general settings.
    :type parameters: Parameters
    :param plot_iterations: If true, plot fit results in each iterations
    :type plot_iterations: bool

    :return: ---
    :rtype: None
    """

    # Set name of result and target quantity index
    result_label = parameters.target_label
    target_quantity_idx = parameters.target_label

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
        # Determine start and end date
        start_date = raw_data.loc[idx_end - parameters.num_data_points_gp_fit + 1]["Date"]
        end_date = raw_data.loc[idx_end]["Date"]

        if plot_iterations:
            prediction_mode = "all"

        else:
            prediction_mode = "predict_only"

        result = fit_gp(raw_data,
                        parameters,
                        subsample_timeframe=False,
                        start_date=start_date,
                        end_date=end_date,
                        prediction_horizon=1,
                        prediction_horizon_mode="index",
                        prediction_mode=prediction_mode,
                        plot_results=plot_iterations)

        # calculate prediction and store
        if parameters.use_return:
            prediction_error[i] = result[result_label].values[-1] - \
                                  raw_data[target_quantity_idx][idx_end + 1]
            martingale_error[i] = -raw_data[target_quantity_idx][idx_end + 1]
        else:
            prediction_error[i] = result[result_label].values[-1] - \
                                  raw_data[target_quantity_idx][idx_end + 1]
            martingale_error[i] = raw_data[target_quantity_idx][idx_end] - \
                                  raw_data[target_quantity_idx][idx_end + 1]

    plot_prediction_error_statistic(prediction_error,
                                    reference_error=martingale_error,
                                    num_bins=parameters.histogram_num_bins)
