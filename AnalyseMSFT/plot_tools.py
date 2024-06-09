import matplotlib.pyplot as plt
import matplotlib.dates as pltdates
import numpy as np
import pandas as pd


def plot_prediction_result(train_data,
                           test_data,
                           result,
                           target_quantity_idx,
                           result_idx="prediction",
                           plot_shading_mode=2.0):
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

    plt.title("GP fit for quantity "+result_idx)
    plt.xlabel("Date")
    plt.ylabel(target_quantity_idx)
    plt.legend(loc='upper right')


    # Set the date format on the x-axis
    ax.xaxis.set_major_formatter(pltdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(pltdates.DayLocator(interval=10))  # Major ticks every 10 days
    ax.xaxis.set_minor_locator(pltdates.DayLocator())

    plt.show()


def plot_prediction_error_statistic(prediction_error, reference_error=None, num_bins=50):

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
