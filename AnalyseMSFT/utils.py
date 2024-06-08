import pandas as pd
from sklearn.model_selection import train_test_split as train_test_split_sklearn

from kernel_functions import rbf_kernel


def load_msft():
    # Load data
    raw_data = pd.read_csv("../DataSets/MSFT.csv")
    # Extract timestamps
    time = pd.to_datetime(raw_data["Date"], format="%Y-%m-%d")
    # Convert timestamps to days from start
    delta_time = (time - time[0]).dt.days
    # Add column delta_time to raw_data
    raw_data["dt"] = delta_time

    return raw_data


def select_data_time(data_full, start_date, end_date, date_idx="Date"):
    idx = (data_full[date_idx] >= start_date) & (data_full[date_idx] <= end_date)
    data_subset = data_full[idx]
    return data_subset

def train_test_split(data, test_data_size):
    train_data, test_data = train_test_split_sklearn(data, test_size=test_data_size, random_state=42)
    train_data = train_data.sort_index()
    test_data = test_data.sort_index()

    return train_data, test_data


def compute_kernel_matrices(predict_data, train_data, rbf_length_scale, rbf_output_scale, kernel_type="rbf"):
    if kernel_type == "rbf":
        k_xx = rbf_kernel(train_data, train_data, length_scale=rbf_length_scale, output_scale=rbf_output_scale)
        k_zx = rbf_kernel(predict_data, train_data, length_scale=rbf_length_scale, output_scale=rbf_output_scale)
        k_zz = rbf_kernel(predict_data, predict_data, length_scale=rbf_length_scale, output_scale=rbf_output_scale)
    else:
        return

    return k_xx, k_zx, k_zz


def construct_prediction_result(data, mu_predicted, std_predicted, result_label="prediction"):
    # Create result series
    result = pd.DataFrame({
        "Date": data["Date"].reset_index(drop=True),
        "dt": data["dt"].reset_index(drop=True),
        result_label: pd.Series(mu_predicted.reshape(-1)),
        "std": pd.Series(std_predicted)
    })
    return result


def compute_return(data, target_price_idx):

    data["Return"] = 100.0 * data[target_price_idx].diff()/data[target_price_idx].shift(1)
    return data.dropna(subset=["Return"])
