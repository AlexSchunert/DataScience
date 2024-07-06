import pandas as pd
import numpy as np
from gp_main import gp_process
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle


def compute_acf(time, signal, max_lag=None):
    """
    Computes acf of signal. Irregular sampling and missing data is handled using time vector

    :param time: The timestamps of the signal values
    :type time: ndarray
    :param signal: The signal values for which acf is calculated
    :type signal: ndarray
    :param max_lag: Max lag in days
    :type max_lag: float

    :return: Tuple of lag and autocovariance
    :rtype: tuple(ndarray,ndarray)
    """
    if max_lag is None:
        max_lag = time.max() - time.min()

    frequency, power = LombScargle(time, signal).autopower(maximum_frequency=0.5 / np.median(np.diff(time)))

    # Convert the power spectral density to the autocorrelation function
    acf = np.fft.irfft(power)
    lags = np.fft.fftfreq(len(acf), d=(frequency[1] - frequency[0]))
    idx = np.where((lags >= 0) & (lags <= max_lag))[0]
    acf = acf[idx]
    lags = lags[idx]

    return lags, acf


def fit_acf(dt, correlation):
    """
    Fit functions to acf to determine hyperparameters of kernel functions

    :param dt: The lag (=> x)
    :type dt: ndarray
    :param correlation: The autocorrelation (=> y)
    :type correlation: ndarray

    :return: Tuple of (Array of optimized function parameters, Function values for dt)
    :rtype: (ndarray,ndarray)
    """
    target_quantity_idx = "acf"
    result_label = "acf"
    test_data = pd.DataFrame({
        "dt": dt,
        "acf": correlation
    })
    sigma_measurement = 0.05
    rbf_length_scale = 180.0
    rbf_output_scale = 1.0
    gp_result, gp_posterior = gp_process(test_data,
                                         test_data,
                                         target_quantity_idx,
                                         result_label,
                                         sigma_measurement,
                                         rbf_length_scale,
                                         rbf_output_scale)

    # Exponential
    # acf_model = lambda x, x0, l: np_exp(-abs(x-x0) / l)
    # fit_result = curve_fit(acf_model, dt, correlation, p0=(0.0, 10.0))
    # pfit_opt = fit_result[0]
    # fct_values = acf_model(dt, pfit_opt[0],pfit_opt[1])

    # Exponential + periodic
    # acf_model = lambda x, x0, l, A, f, phi0: np_exp(-abs(x - x0) / l) + A * sin(f*x + phi0)
    # fit_result = curve_fit(acf_model, dt, correlation, p0=(0.0, 10.0, 0.1, 0.0, 0.0))
    # pfit_opt = fit_result[0]
    # fct_values = acf_model(dt, pfit_opt[0], pfit_opt[1], pfit_opt[2], pfit_opt[3], pfit_opt[4])

    # periodic
    # acf_model = lambda x, A, f, phi0: A * sin(f * x + phi0)
    # fit_result = curve_fit(acf_model, dt, correlation, p0=(1.0, 0.02, 0.0))
    # pfit_opt = fit_result[0]
    # fct_values = acf_model(dt, pfit_opt[0], pfit_opt[1], pfit_opt[2])

    # hyperbolic
    # acf_model = lambda x, A, x0, y0: A / abs(x + x0) + y0
    # fit_result = curve_fit(acf_model, dt, correlation, p0=(1.0, 1.0, 0.0))
    # pfit_opt = fit_result[0]
    # fct_values = acf_model(dt, pfit_opt[0], pfit_opt[1], pfit_opt[2])
    # print(pfit_opt)

    # pfit_opt = []
    # fct_values = gp_result["acf"]
    return gp_result, gp_posterior
