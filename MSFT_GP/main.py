from dataclasses import dataclass
# from numpy import fft, abs
from numpy.random import randn
from scipy.fft import fft, fftfreq
from scipy.signal import windows, periodogram, correlate
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from utils import load_msft, Parameters, select_data_time
from analyses import fit_gp, gp_prediction_vs_martingale
from plot_tools import plot_data, plot_sliding_window_autocorr


def run_initial_example():
    # Parameters
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
    # Fit GP and plot
    fit_gp(raw_data, parameters, prediction_horizon_mode="day", subsample_timeframe=True, prediction_mode="all")

def plot_return_ts():
    # Parameters
    parameters = Parameters(start_date="1980-01-01",
                            end_date="2024-12-31",
                            tick_interval_x=1000,
                            use_return=True,
                            prediction_horizon=10,
                            target_label="Adj Close")
    # Load dataset
    raw_data = load_msft(parameters)
    raw_data = select_data_time(raw_data, parameters.start_date, parameters.end_date)
    # Plot
    plot_data(raw_data,
              "Return",
              "Date",
              plot_format="b",
              title="One-day returns from adjusted closing stock price",
              mode="Standard",
              tick_interval_x=parameters.tick_interval_x)



def plot_return_full():
    # Parameters
    parameters = Parameters(start_date="1980-01-01",
                            end_date="2024-12-31",
                            tick_interval_x=1000,
                            use_return=True,
                            prediction_horizon=10,
                            target_label="Adj Close")
    # Load dataset
    raw_data = load_msft(parameters)
    raw_data = select_data_time(raw_data, parameters.start_date, parameters.end_date)
    # Plot
    plot_data(raw_data,
              "Return",
              "Date",
              plot_format="b",
              title="",
              mode="Full",
              tick_interval_x=parameters.tick_interval_x,
              nlag_acf=360)

def plot_return_full_subs():
    # Parameters
    parameters_low = Parameters(start_date="1993-01-01",
                                end_date="1995-12-31",
                                tick_interval_x=1000,
                                use_return=True,
                                prediction_horizon=10,
                                target_label="Adj Close")

    parameters_high = Parameters(start_date="2000-01-01",
                                 end_date="2003-12-31",
                                 tick_interval_x=1000,
                                 use_return=True,
                                 prediction_horizon=10,
                                 target_label="Adj Close")

    # Load dataset
    raw_data = load_msft(parameters_low)
    data_subs_low = select_data_time(raw_data, parameters_low.start_date, parameters_low.end_date)
    data_subs_high = select_data_time(raw_data, parameters_high.start_date, parameters_high.end_date)

    # Plot
    plot_data(data_subs_low,
              "Return",
              "Date",
              plot_format="b",
              title="",
              mode="Full",
              tick_interval_x=parameters_low.tick_interval_x,
              nlag_acf=360)

    plot_data(data_subs_high,
              "Return",
              "Date",
              plot_format="b",
              title="",
              mode="Full",
              tick_interval_x=parameters_high.tick_interval_x,
              nlag_acf=360)

def make_arg_parser():
    """
    Create ArgumentParser object

    :return: The created ArgumentParser object.
    :rtype: ArgumentParser
    """

    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, help="Mode: init_example/plot_return_ts/plot_return_full/plot_return_full_subs", required=False)

    return parser


def main():
    # create parser
    parser = make_arg_parser()
    # parse command line
    args = parser.parse_args()

    if args.mode is None:
        mode = "plot_return_full"
    else:
        mode = args.mode

    if mode == "init_example":
        run_initial_example()
    elif mode == "plot_return_ts":
        plot_return_ts()
    elif mode == "plot_return_full":
        plot_return_full()
    elif mode == "plot_return_full_subs":
        plot_return_full_subs()
    else:
        print("Invalid mode")

    # fit_gp(raw_data, parameters, prediction_horizon_mode="day", subsample_timeframe=True, prediction_mode="all")
    # plot_sliding_window_autocorr(raw_data, "Return", "dt", window_size=180)


if __name__ == '__main__':
    main()
