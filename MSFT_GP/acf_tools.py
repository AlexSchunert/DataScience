import pandas as pd
import numpy as np
from gp_main import gp_process
from gp_lib import GPPosterior
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle


def compute_gp_kernel_posterior(data,
                                data_label_y):
    """
    Computes acf and fits gp. Additionally, estimates std of data noise. This i

    :param data: DataFrame containing columns with labels data_label_y and "dt"
    :type data: pd.DataFrame
    :param data_label_y: String label of y-axis data
    :type data_label_y: str

    :return: Tuple of GPPosterior and estimated noise std
    :rtype: (GPPosterior, float)
    """

    signal = data[data_label_y].values
    # Get signals and time
    t = data["dt"].values
    # Autocovariance
    lag_acf, auto_cov = compute_acf(t, signal)
    # We need to rescale such that acf[0] = var(signal)
    auto_cov = auto_cov / auto_cov[0] * signal.var()
    # Fit gp_kernel to acf
    gp_result, gp_posterior = fit_acf(lag_acf, auto_cov)

    # Estimate noise std
    # Since m_i = s_i + n_i => E[m_i*m_(i+k)]=E[s_i*s_(i+k)]+E[s_i*n_(i+k)]+E[s_(i+k)*n_i]E[n_i*n_(i+k)]
    # For k=0 => E[m_i*m_i]=E[s_i*s_i]+E[n_i*n_i]=sigma_s^2+sigma_n^2
    # E[m_i*m_i]
    var_m = signal.var()
    # variance due to signal E[s_i*s_i] -> We use the value estimated by the gp_kernel as estimate => assuming a somehow
    # smooth acf
    var_s = gp_result.loc[gp_result["dt"] == 0, "acf"].values[0]
    # variance due to noise
    var_n = var_m - var_s
    std_n = np.sqrt(var_n)

    return gp_posterior, std_n

def compute_acf(time, signal, max_lag=None):
    """
    Computes acf of signal. Irregular sampling and missing data is handled using time vector

    :param time: The timestamps of the signal values
    :type time: np.ndarray
    :param signal: The signal values for which acf is calculated
    :type signal: np.ndarray
    :param max_lag: Max lag in days
    :type max_lag: float

    :return: Tuple of lag and autocovariance
    :rtype: tuple(np.ndarray,np.ndarray)
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
    :type dt: np.ndarray
    :param correlation: The autocorrelation (=> y)
    :type correlation: np.ndarray

    :return: Tuple of (GPResult, GPPosterior)
    :rtype: (pd.DataFrame,GPPosterior)
    """
    target_quantity_idx = "acf"
    result_label = "acf"
    test_data = pd.DataFrame({
        "dt": dt,
        "acf": correlation
    })
    sigma_measurement = 0.05
    rbf_length_scale = 45.0
    rbf_output_scale = 1.0
    gp_result, gp_posterior = gp_process(test_data,
                                         test_data,
                                         target_quantity_idx,
                                         result_label,
                                         sigma_measurement,
                                         rbf_length_scale=rbf_length_scale,
                                         rbf_output_scale=rbf_output_scale)

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
