import matplotlib.pyplot as plt
import matplotlib.dates as pltdates
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import windows, correlate, periodogram
from statsmodels.tsa.stattools import acovf
from astropy.timeseries import LombScargle
from utils import autocorrelations_sliding_window
from acf_tools import fit_acf

def plot_prediction_result(train_data,
                           test_data,
                           result,
                           target_quantity_idx,
                           result_idx="prediction",
                           plot_shading_mode="2-sigma",
                           tick_interval_x=10):
    """
    Plots the result of a gp prediction.

    :param train_data: DataFrame with data used for training. Must contain columns labeled "Date" and
                       target_quantity_idx
    :type train_data: pd.DataFrame
    :param test_data: DataFrame with data used for testing/prediction. Must contain columns labeled "Date" and
                      target_quantity_idx
    :type test_data: pd.DataFrame
    :param result: DataFrame with gp prediction results. Must contain columns labeled "Date" and result_idx
    :type result: pd.DataFrame
    :param target_quantity_idx: Label of target quantity used in gp prediction
    :type target_quantity_idx: str
    :param result_idx: Label of gp prediction result
    :type result_idx: str
    :param plot_shading_mode: Determines the region around gp mean function that is shaded. Currently only "2-sigma"
                              is supported => 2-sigma interval around gp mean is shaded in gray.
    :type plot_shading_mode: str
    :param tick_interval_x: Tick on x-axis every tick_interval_x days
    :type tick_interval_x: int

    :return: ---
    :rtype: None
    """

    fig, ax = plt.subplots()
    # Plot data
    plt.plot(pd.to_datetime(test_data["Date"]), test_data[target_quantity_idx], 'b.', label="Test data")
    plt.plot(pd.to_datetime(train_data["Date"]), train_data[target_quantity_idx], 'g*', label="Train data")
    # Plot prediction
    plt.plot(pd.to_datetime(result["Date"]), result[result_idx], 'g-', label="GP mean-fct.")
    # Plot standard deviation
    plt.plot(pd.to_datetime(result["Date"]), result[result_idx] + result["std"], 'r--')
    plt.plot(pd.to_datetime(result["Date"]), result[result_idx] - result["std"], 'r--', label="1-sigma")
    if plot_shading_mode == "2-sigma":
        upper_bound = result[result_idx] + 2.0 * result["std"]
        lower_bound = result[result_idx] - 2.0 * result["std"]
        plt.fill_between(pd.to_datetime(result["Date"]), lower_bound, upper_bound, where=(upper_bound >= lower_bound),
                         interpolate=True,
                         color='gray', alpha=0.5, label=plot_shading_mode)
    else:
        upper_bound = result[result_idx] + 2.0 * result["std"]
        lower_bound = result[result_idx] - 2.0 * result["std"]
        plt.fill_between(pd.to_datetime(result["Date"]), lower_bound, upper_bound, where=(upper_bound >= lower_bound),
                         interpolate=True,
                         color='gray', alpha=0.5, label=plot_shading_mode)

    plt.title("GP fit for quantity " + result_idx)
    plt.xlabel("Date")
    plt.ylabel(target_quantity_idx)
    plt.legend(loc='upper left')

    # Set the date format on the x-axis
    ax.xaxis.set_major_formatter(pltdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(pltdates.DayLocator(interval=tick_interval_x))  # Major ticks every 10 days
    ax.xaxis.set_minor_locator(pltdates.DayLocator(interval=tick_interval_x))

    plt.show()


def plot_prediction_error_statistic(prediction_error, reference_error=None, num_bins=50):
    """
    Plots histograms of prediction_error and reference_error

    :param prediction_error: Error of gp prediction
    :type prediction_error: np.ndarray
    :param reference_error: Error of reference prediction (e.g. constant stock price)
    :type reference_error: np.ndarray
    :param num_bins: Number of histogram bins
    :type num_bins: int

    :return: ---
    :rtype: None
    """

    # Compute mean and std for error vecs, create label, plot
    mean_prediction_error = np.mean(prediction_error)
    std_prediction_error = np.std(prediction_error)
    label_prediction_error = \
        "Prediction error, m: " + str(round(mean_prediction_error, 2)) + ", s: " + str(round(std_prediction_error, 2))

    plt.figure()
    plt.hist(prediction_error, bins=num_bins, color="green", histtype="bar", alpha=0.5, rwidth=0.8, density=True,
             label=label_prediction_error)

    if reference_error is not None:
        mean_reference_error = np.mean(reference_error)
        std_reference_error = np.std(reference_error)
        label_reference_error = \
            "Reference error, m: " + str(round(mean_reference_error, 2)) + ", s: " + str(round(std_reference_error, 2))
        plt.hist(reference_error, bins=num_bins, color="gray", histtype="bar", alpha=0.5, rwidth=0.8, density=True,
                 label=label_reference_error)

    plt.legend(loc='upper right', title='Histograms')
    plt.xlabel("error")
    plt.ylabel("f")
    plt.show()

def plot_data(data,
              data_label_y,
              data_label_x="Date",
              plot_format=".",
              title="",
              mode="Standard",
              nbins=100,
              tick_interval_x=10,
              nlag_acf=180):
    """
    Two-dimensional plot of data columns identified by data_label_y and data_label_x. If mode=="Full", also
    periodogram, histogram, and estimated autocorrelation are shown
    :param data: DataFrame containing columns with labels data_label_y and data_label_x
    :type data: pd.DataFrame
    :param data_label_y: String label of y-axis data
    :type data_label_y: str
    :param data_label_x: String label of x-axis data
    :type data_label_x: str
    :param plot_format: Format string passed to matplotlibs plot fct.
    :type plot_format: str
    :param title: Title of the plot
    :type title: str
    :param mode: "Standard" for simple timeseries plot, "Full" for timeseries, periodogram, histogram,
                 and estimated autocorrelation
    :type mode: str
    :param nbins: Number of histogram bins in mode == "Full"
    :type nbins: int
    :param tick_interval_x: Tick on x-axis every tick_interval_x days. Currently not used.
    :type tick_interval_x: int
    :param nlag_acf: Max lag for autocorrelation fct
    :type nlag_acf: int

    :return: ---
    :rtype: None
    """

    if mode == "Standard":
        fig, ax = plt.subplots(1, 1)
        ax.plot(pd.to_datetime(data[data_label_x]), data[data_label_y], plot_format)
        ax.set_xlabel(data_label_x)
        ax.set_ylabel(data_label_y)
        ax.set_title(title)
        # Set the date format on the x-axis
        plt.show()

    if mode == "Full":
        signal = data[data_label_y].values  # 0.1 * randn(raw_data["Return"].values.shape[0])

        # Get signals and time
        t = data["dt"].values
        signal = signal
        # Windowing for spectrum calculation
        window = windows.hamming(len(signal))
        windowed_signal = signal * window
        # Spectrum
        # f, Pxx = periodogram(windowed_signal, 1)
        # Compute Lomb-Scargle periodogram
        f, Pxx = LombScargle(t.astype(np.float64), signal).autopower(maximum_frequency=0.5 / np.median(np.diff(t)))
        # Autocovariance
        auto_cov = acovf(pd.Series(signal, index=t), missing="drop", nlag=nlag_acf)
        # auto_cov = correlate(signal, signal, mode="full")[len(signal) - 1:] / len(signal)
        auto_cov = auto_cov / auto_cov[0]
        # Plot
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(data["dt"].values, signal, plot_format)
        axs[0, 0].set_xlabel("time")
        axs[0, 0].set_ylabel("signal")
        axs[0, 0].set_title("signal vs time")

        axs[1, 0].plot(f, Pxx, plot_format)
        axs[1, 0].set_xlabel("frequency")
        axs[1, 0].set_ylabel("power")
        axs[1, 0].set_title("PSD")
        axs[0, 1].plot(auto_cov, plot_format)
        axs[0, 1].set_xlabel("dt")
        axs[0, 1].set_ylabel("Corr")
        axs[0, 1].set_title("Autocorrelation")
        axs[1, 1].hist(signal, bins=nbins, color="blue")
        axs[1, 1].set_xlabel(data_label_y)
        axs[1, 1].set_ylabel("frequency")
        axs[1, 1].set_title("Histogram")
        plt.tight_layout()
        plt.show()


def plot_sliding_window_autocorr(data,
                                 data_label_y,
                                 data_label_x="dt",
                                 window_size=180):
    """
    Calculate autocorrelation with sliding window and plot as 2.5D

    :param data: DataFrame containing columns with labels data_label_y and data_label_x
    :type data: pd.DataFrame
    :param data_label_y: String label of y-axis data
    :type data_label_y: str
    :param data_label_x: String label of x-axis data
    :type data_label_x: str
    :param window_size: Length of autocorrelation window.
    :type window_size: ndarray

    :return: ---
    :rtype: None
    """
    signal = data[data_label_y].values  # 0.1 * randn(raw_data["Return"].values.shape[0])

    # Get signals and time
    t = data["dt"]
    signal = signal
    autocorr = autocorrelations_sliding_window(signal, window_size=window_size)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface
    x, y = np.meshgrid(np.arange(autocorr.shape[1]), np.arange(autocorr.shape[0]))
    surf = ax.plot_surface(x, y, autocorr, cmap='viridis')
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def plot_acf_fit(data,
                 data_label_y,
                 title="",
                 nlag_acf=180):
    """
    Calculate acf of data, fit a function, and plot the result

    :param data: DataFrame containing columns with labels data_label_y and data_label_x
    :type data: pd.DataFrame
    :param data_label_y: String label of y-axis data
    :type data_label_y: str
    :param title: Title of the plot
    :type title: str
    :param nlag_acf: Max lag for autocorrelation fct
    :type nlag_acf: int

    :return: ---
    :rtype: None
    """

    signal = data[data_label_y].values  # 0.1 * randn(raw_data["Return"].values.shape[0])
    # Get signals and time
    t = data["dt"].values

    # Autocovariance
    auto_cov = acovf(pd.Series(signal, index=t), missing="drop", nlag=nlag_acf)
    # auto_cov = correlate(signal, signal, mode="full")[len(signal) - 1:] / len(signal)
    auto_cov = auto_cov / auto_cov[0]
    lag_acf = np.arange(auto_cov.shape[0])
    pfit_opt, fct_values = fit_acf(lag_acf, auto_cov)

    plt.plot(lag_acf, auto_cov,'b')
    plt.plot(lag_acf, fct_values, 'r')

    plt.show()