import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel


# from time import mktime
# from datetime import datetime as dt

def select_data_time(data_full, start_date, end_date):
    idx = (data_full['Date'] >= start_date) & (data_full['Date'] <= end_date)
    data_subset = data_full[idx]
    return data_subset


# Parameters
test_data_size = 0.5
rbf_gamma = 0.01
rbf_amp = 15.0
sigma_price = 0.2  # Make sigma_price a function of time?
target_price = "High"
plot_shading_mode = "2-sigma"

# Load data
raw_data = pd.read_csv("../DataSets/MSFT.csv")
# Extract timestamps
time = pd.to_datetime(raw_data["Date"], format="%Y-%m-%d")
# Convert timestamps to days from start
delta_time = (time - time[0]).dt.days
# Add column delta_time to raw_data
raw_data["dt"] = delta_time

# Create data to fit
# data = pd.DataFrame({
#    "Date": raw_data["Date"],
#    "dt": delta_time,
#    "price": raw_data["High"]
# })

# Split into train- and test-data
data = select_data_time(raw_data, "2000-01-01", "2005-12-31")
data = data[["Date", "dt", target_price]]

train_data, test_data = train_test_split(data, test_size=test_data_size, random_state=42)
train_data = train_data.sort_index()
test_data = test_data.sort_index()

# train_data = select_data_time(train_data, "2000-01-01", "2005-12-31")
# test_data = select_data_time(test_data, "2000-01-01", "2005-12-31")

# Compute kernel for data points => K_xx
k_xx = rbf_amp*rbf_kernel(train_data['dt'].values.reshape(-1, 1), train_data['dt'].values.reshape(-1, 1), gamma=rbf_gamma)
# Construct predictive covariance
## Compute Cholesky decomposition
L = np.linalg.cholesky(k_xx + sigma_price ** 2 * np.eye(k_xx.shape[0]))
## Compute inverse => predictive covariance
predictive_cov = np.linalg.inv(L.T).dot(np.linalg.inv(L))

# Compute representer weights => zero prior
alpha = predictive_cov @ train_data[target_price].values.reshape(-1, 1)

# predict for all data
k_zx = rbf_amp*rbf_kernel(data['dt'].values.reshape(-1, 1), train_data['dt'].values.reshape(-1, 1), gamma=rbf_gamma)
k_zz = rbf_amp*rbf_kernel(data['dt'].values.reshape(-1, 1), data['dt'].values.reshape(-1, 1), gamma=rbf_gamma)
## prediction
price_predicted = k_zx @ alpha
## stdev of prediction
cov_prediction = (k_zz + sigma_price ** 2 * np.eye(k_zz.shape[0])) - k_zx @ predictive_cov @ k_zx.T
std_prediction = np.sqrt(np.diagonal(cov_prediction))

# Create result series
result = pd.DataFrame({
    "dt": data["dt"].reset_index(drop=True),
    "price_prediction": pd.Series(price_predicted.reshape(-1)),
    "std": pd.Series(std_prediction)
})

plt.plot(train_data["dt"], train_data[target_price], 'k+')
plt.plot(test_data["dt"], test_data[target_price], 'b.')
plt.plot(result["dt"], result["price_prediction"], 'g')
plt.plot(result["dt"], result["price_prediction"] + result["std"], 'r--')
plt.plot(result["dt"], result["price_prediction"] - result["std"], 'r--')
if plot_shading_mode == "2-sigma":
    upper_bound = result["price_prediction"] + 2.0 * result["std"]
    lower_bound = result["price_prediction"] - 2.0 * result["std"]
    plt.fill_between(result["dt"], lower_bound, upper_bound, where=(upper_bound >= lower_bound), interpolate=True,
                     color='gray', alpha=0.5)
else:
    upper_bound = result["price_prediction"] + 2.0 * result["std"]
    lower_bound = result["price_prediction"] - 2.0 * result["std"]
    plt.fill_between(result["dt"], lower_bound, upper_bound, where=(upper_bound >= lower_bound), interpolate=True,
                     color='gray', alpha=0.5)

plt.show()

# Split into train and test data
# delta_time_train

# intra_day_diff = raw_data["Open"] - raw_data["Close"]
# intra_day_spread = raw_data["High"] - raw_data["Low"]
# raw_data_plot = raw_data.plot("Date", ["Open","High"])


# fig, ax = plt.subplots(3)
# ax[0].plot(time, raw_data["High"],'.')
# ax[1].plot(time, intra_day_diff)
# ax[2].plot(time, intra_day_diff/raw_data["High"])
# ax[1].plot(time, intra_day_spread)
# ax[2].plot(time, intra_day_spread/raw_data["High"])

# plt.plot(time, raw_data["High"])
# plt.plot(time, intra_day_diff)
# plt.plot(raw_data["High"],intra_day_diff,'.')
plt.show()

# plt.plot(time,raw_data["Open"])
# plt.show()

pass
