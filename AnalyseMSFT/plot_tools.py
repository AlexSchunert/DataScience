import matplotlib.pyplot as plt
import pandas as pd


def plot_prediction_result(train_data, test_data, result, target_price, plot_shading_mode):
    plt.figure()
    plt.plot(pd.to_datetime(train_data["Date"]), train_data[target_price], 'g*')
    plt.plot(pd.to_datetime(test_data["Date"]), test_data[target_price], 'b.')
    plt.plot(pd.to_datetime(result["Date"]), result["price_prediction"], 'g')
    plt.plot(pd.to_datetime(result["Date"]), result["price_prediction"] + result["std"], 'r--')
    plt.plot(pd.to_datetime(result["Date"]), result["price_prediction"] - result["std"], 'r--')
    if plot_shading_mode == "2-sigma":
        upper_bound = result["price_prediction"] + 2.0 * result["std"]
        lower_bound = result["price_prediction"] - 2.0 * result["std"]
        plt.fill_between(pd.to_datetime(result["Date"]), lower_bound, upper_bound, where=(upper_bound >= lower_bound),
                         interpolate=True,
                         color='gray', alpha=0.5)
    else:
        upper_bound = result["price_prediction"] + 2.0 * result["std"]
        lower_bound = result["price_prediction"] - 2.0 * result["std"]
        plt.fill_between(pd.to_datetime(result["Date"]), lower_bound, upper_bound, where=(upper_bound >= lower_bound),
                         interpolate=True,
                         color='gray', alpha=0.5)

    plt.show()
