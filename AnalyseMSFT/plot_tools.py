import matplotlib.pyplot as plt
import pandas as pd


def plot_prediction_result(train_data,
                           test_data,
                           result,
                           target_quantity_idx,
                           result_idx="prediction",
                           plot_shading_mode=2.0):
    plt.figure()
    plt.plot(pd.to_datetime(train_data["Date"]), train_data[target_quantity_idx], 'g*')
    plt.plot(pd.to_datetime(test_data["Date"]), test_data[target_quantity_idx], 'b.')
    plt.plot(pd.to_datetime(result["Date"]), result[result_idx], 'g')
    plt.plot(pd.to_datetime(result["Date"]), result[result_idx] + result["std"], 'r--')
    plt.plot(pd.to_datetime(result["Date"]), result[result_idx] - result["std"], 'r--')
    if plot_shading_mode == "2-sigma":
        upper_bound = result[result_idx] + 2.0 * result["std"]
        lower_bound = result[result_idx] - 2.0 * result["std"]
        plt.fill_between(pd.to_datetime(result["Date"]), lower_bound, upper_bound, where=(upper_bound >= lower_bound),
                         interpolate=True,
                         color='gray', alpha=0.5)
    else:
        upper_bound = result[result_idx] + 2.0 * result["std"]
        lower_bound = result[result_idx] - 2.0 * result["std"]
        plt.fill_between(pd.to_datetime(result["Date"]), lower_bound, upper_bound, where=(upper_bound >= lower_bound),
                         interpolate=True,
                         color='gray', alpha=0.5)

    plt.show()


def plot_prediction_error_statistic(prediction_error, reference_error=None, num_bins=50):
    plt.figure()
    plt.hist(prediction_error, bins=num_bins, color="green", histtype="bar", alpha=0.5, rwidth=0.8, density=True,
             label="Prediction error")
    if reference_error is not None:
        plt.hist(reference_error, bins=num_bins, color="gray", histtype="bar", alpha=0.5, rwidth=0.8, density=True,
                 label="Reference error")

    plt.legend(loc='upper right', title='Histograms')
    plt.xlabel("error")
    plt.ylabel("f")
    plt.show()
