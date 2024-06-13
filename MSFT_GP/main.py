from dataclasses import dataclass
# from numpy import fft, abs
from numpy.random import randn
from scipy.fft import fft, fftfreq
from scipy.signal import windows, periodogram, correlate
from matplotlib import pyplot as plt
from utils import load_msft, Parameters, select_data_time
from analyses import fit_gp, gp_prediction_vs_martingale
from plot_tools import plot_data
import numpy as np


def autocovariance_sliding_window(time_series, window_size):
    n = len(time_series)
    autocovariances = np.zeros((n - window_size + 1, window_size))

    for i in range(n - window_size + 1):
        window = time_series[i:i + window_size]
        autocorr = correlate(window, window, mode='full') / window_size

        # Extract the relevant part of the autocorrelation (positive lags)
        autocorr = autocorr[window_size - 1:window_size + window_size - 1]
        autocorr = autocorr / autocorr[0]
        autocovariances[i, :] = autocorr

    return autocovariances


"""
# Example usage
np.random.seed(0)
time_series = np.random.randn(100)
window_size = 10
print(f'Autocovariances calculated for each window position and lag:')
print(autocovariances)
print(f'Shape of autocovariances array: {autocovariances.shape}')
"""


def run_initial_example():
    parameters = Parameters(num_data_points_gp_fit=50,
                            start_date="1989-06-01",
                            end_date="1989-12-31",
                            test_data_size=0.2,
                            rbf_length_scale=10.0,
                            rbf_output_scale=5.0,
                            tick_interval_x=50,
                            use_return=False,
                            sigma_price=0.01,
                            prediction_horizon=0)
    # Load dataset
    raw_data = load_msft(parameters)
    fit_gp(raw_data, parameters, prediction_horizon_mode="day", subsample_timeframe=True, prediction_mode="all")


parameters = Parameters(num_data_points_gp_fit=50,
                        start_date="1980-01-01",
                        end_date="2024-12-31",
                        test_data_size=0.01,
                        rbf_length_scale=1.0,
                        rbf_output_scale=5.0,
                        tick_interval_x=50,
                        use_return=True,
                        sigma_return=0.001,
                        prediction_horizon=10,
                        target_label="High")

raw_data = load_msft(parameters)
raw_data = select_data_time(raw_data, parameters.start_date, parameters.end_date)
# fit_gp(raw_data, parameters, prediction_horizon_mode="day", subsample_timeframe=True, prediction_mode="all")
plot_data(raw_data, "Return", "Date", plot_format="r", title="Plot highest stock price")

signal = raw_data["Return"].values**2   # 0.1 * randn(raw_data["Return"].values.shape[0])
print(signal.mean())
print(signal.std())
# signal = signal - signal.mean()
t = raw_data["dt"]
signal = signal
window = windows.hamming(len(signal))
windowed_signal = signal * window
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(t, signal)

f, Pxx = periodogram(windowed_signal, 1)
axs[1, 0].plot(f, Pxx)

auto_cov = correlate(signal, signal, mode="full")[len(signal) - 1:] / len(signal)
auto_cov = auto_cov / auto_cov[0]
axs[0, 1].plot(auto_cov)

axs[1, 1].hist(signal, bins=100)
print()

# frequencies = fftfreq(len(auto_cov), d=1)
# axs[2].plot(frequencies[:len(frequencies) // 2], abs(auto_cov)[:len(frequencies) // 2])
# axs[2].plot(t[:len(t) // 2], abs(auto_cov)[:len(t) // 2])

plt.tight_layout()
plt.show()


lenght_cov = 180
autocovariances = autocovariance_sliding_window(signal, lenght_cov)

"""
plt.imshow(autocovariances[1:, :], cmap="jet")
plt.show()
"""


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the surface
X, Y = np.meshgrid(np.arange(autocovariances.shape[1]),np.arange(autocovariances.shape[0]))
surf = ax.plot_surface(X, Y, autocovariances, cmap='viridis')
plt.show()

pass

# Parameters


# Process stuff
# plot_data(raw_data, "High", "Date", plot_format="r", title="Plot highest stock price")

# gp_prediction_vs_martingale(raw_data, parameters, plot_iterations=True)

pass
