from dataclasses import dataclass
# from numpy import fft, abs
from numpy.random import randn
from scipy.fft import fft, fftfreq
from scipy.signal import windows, periodogram, correlate
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from utils import load_msft, Parameters, select_data_time
from analyses import fit_gp, gp_prediction_vs_martingale
from plot_tools import plot_data, plot_sliding_window_autocorr, plot_acf_fit


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
    fit_gp(raw_data,
           parameters,
           prediction_horizon_mode="day",
           subsample_timeframe=True,
           prediction_mode="all",
           complete=False)


def plot_return_ts(return_mode="standard"):
    """
    Only timeseries plot of target quantity, complete timeframe

    :param return_mode: "If "standard", the signed returns are used. If "abs" absolute values of returns are used"
    :type return_mode: str

    :return: ---
    :rtype: None
    """
    # Parameters
    parameters = Parameters(start_date="1980-01-01",
                            end_date="2024-12-31",
                            tick_interval_x=1000,
                            use_return=True,
                            prediction_horizon=10,
                            target_label="Adj Close",
                            return_mode=return_mode)
    # Load dataset
    raw_data = load_msft(parameters)
    raw_data = select_data_time(raw_data, parameters.start_date, parameters.end_date)
    # Plot
    title = "Abs returns from adj. closing price"  # "One-day returns from adjusted closing stock price"
    plot_data(raw_data,
              parameters.target_label,
              "Date",
              plot_format="b",
              title=title,
              mode="Standard",
              tick_interval_x=parameters.tick_interval_x)


def plot_return_full(return_mode="standard"):
    """
    All plots, complete timeframe

    :param return_mode: "If "standard", the signed returns are used. If "abs" absolute values of returns are used"
    :type return_mode: str

    :return: ---
    :rtype: None
    """

    # Parameters
    parameters = Parameters(start_date="1980-01-01",
                            end_date="2024-12-31",
                            tick_interval_x=1000,
                            use_return=True,
                            prediction_horizon=10,
                            target_label="Adj Close",
                            return_mode=return_mode)
    # Load dataset
    raw_data = load_msft(parameters)
    raw_data = select_data_time(raw_data, parameters.start_date, parameters.end_date)
    # Plot
    plot_data(raw_data,
              parameters.target_label,
              "Date",
              plot_format="b",
              title="",
              mode="Full",
              tick_interval_x=parameters.tick_interval_x,
              nlag_acf=360)


def plot_return_full_subs(return_mode="standard"):
    """
    All plots, two subset timeframes

    :param return_mode: "If "standard", the signed returns are used. If "abs" absolute values of returns are used"
    :type return_mode: str

    :return: ---
    :rtype: None
    """

    # Parameters
    parameters_low = Parameters(start_date="1993-01-01",
                                end_date="1995-12-31",
                                tick_interval_x=1000,
                                use_return=True,
                                prediction_horizon=10,
                                target_label="Adj Close",
                                return_mode=return_mode)

    parameters_low_2 = Parameters(start_date="2011-01-01",
                                  end_date="2012-12-31",
                                  tick_interval_x=1000,
                                  use_return=True,
                                  prediction_horizon=10,
                                  target_label="Adj Close",
                                  return_mode=return_mode)

    parameters_high = Parameters(start_date="2000-01-01",
                                 end_date="2003-12-31",
                                 tick_interval_x=1000,
                                 use_return=True,
                                 prediction_horizon=10,
                                 target_label="Adj Close",
                                 return_mode=return_mode)

    parameters_high_2 = Parameters(start_date="2008-01-01",
                                   end_date="2009-12-31",
                                   tick_interval_x=1000,
                                   use_return=True,
                                   prediction_horizon=10,
                                   target_label="Adj Close",
                                   return_mode=return_mode)

    # Load dataset
    raw_data = load_msft(parameters_low)
    parameters_low_2.target_label = parameters_low.target_label
    parameters_high.target_label = parameters_low.target_label
    parameters_high_2.target_label = parameters_low.target_label
    data_subs_low = select_data_time(raw_data, parameters_low.start_date, parameters_low.end_date)
    data_subs_low_2 = select_data_time(raw_data, parameters_low_2.start_date, parameters_low_2.end_date)
    data_subs_high = select_data_time(raw_data, parameters_high.start_date, parameters_high.end_date)
    data_subs_high_2 = select_data_time(raw_data, parameters_high_2.start_date, parameters_high_2.end_date)

    # Plot
    plot_data(data_subs_low,
              parameters_low.target_label,
              "Date",
              plot_format="b",
              title="",
              mode="Full",
              tick_interval_x=parameters_low.tick_interval_x,
              nlag_acf=360)

    plot_data(data_subs_low_2,
              parameters_low_2.target_label,
              "Date",
              plot_format="b",
              title="",
              mode="Full",
              tick_interval_x=parameters_low_2.tick_interval_x,
              nlag_acf=360)

    plot_data(data_subs_high,
              parameters_high.target_label,
              "Date",
              plot_format="b",
              title="",
              mode="Full",
              tick_interval_x=parameters_high.tick_interval_x,
              nlag_acf=360)

    plot_data(data_subs_high_2,
              parameters_high_2.target_label,
              "Date",
              plot_format="b",
              title="",
              mode="Full",
              tick_interval_x=parameters_high_2.tick_interval_x,
              nlag_acf=360)


def fit_acf_function_subsets(return_mode="standard"):
    """
    Compute acf, fit, and plot for subsets

    :param return_mode: "If "standard", the signed returns are used. If "abs" absolute values of returns are used"
    :type return_mode: str

    :return: ---
    :rtype: None
    """
    params_lcorr_subs1 = Parameters(start_date="1993-01-01",
                                    end_date="1995-12-31",
                                    tick_interval_x=1000,
                                    use_return=True,
                                    prediction_horizon=10,
                                    target_label="Adj Close",
                                    return_mode=return_mode)

    params_lcorr_subs2 = Parameters(start_date="2011-01-01",
                                    end_date="2012-12-31",
                                    tick_interval_x=1000,
                                    use_return=True,
                                    prediction_horizon=10,
                                    target_label="Adj Close",
                                    return_mode=return_mode)

    params_hcorr_subs1 = Parameters(start_date="2000-01-01",
                                    end_date="2003-12-31",
                                    tick_interval_x=1000,
                                    use_return=True,
                                    prediction_horizon=10,
                                    target_label="Adj Close",
                                    return_mode=return_mode)

    params_hcorr_subs2 = Parameters(start_date="2008-01-01",
                                    end_date="2009-12-31",
                                    tick_interval_x=1000,
                                    use_return=True,
                                    prediction_horizon=10,
                                    target_label="Adj Close",
                                    return_mode=return_mode)

    # Parameters

    # Load dataset
    raw_data = load_msft(params_lcorr_subs1)
    params_lcorr_subs2.target_label = params_lcorr_subs1.target_label
    params_hcorr_subs1.target_label = params_lcorr_subs1.target_label
    params_hcorr_subs2.target_label = params_lcorr_subs1.target_label

    data_lcorr_subs1 = select_data_time(raw_data, params_lcorr_subs1.start_date, params_lcorr_subs1.end_date)
    data_lcorr_subs2 = select_data_time(raw_data, params_lcorr_subs2.start_date, params_lcorr_subs2.end_date)
    data_hcorr_subs1 = select_data_time(raw_data, params_hcorr_subs1.start_date, params_hcorr_subs1.end_date)
    data_hcorr_subs2 = select_data_time(raw_data, params_hcorr_subs2.start_date, params_hcorr_subs2.end_date)

    # Plot
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])

    plot_acf_fit(data_lcorr_subs1,
                 params_lcorr_subs1.target_label,
                 title="TFL",
                 target_axis=ax00)

    plot_acf_fit(data_lcorr_subs2,
                 params_lcorr_subs2.target_label,
                 title="TFL2",
                 target_axis=ax01)

    plot_acf_fit(data_hcorr_subs1,
                 params_hcorr_subs1.target_label,
                 title="TFH",
                 target_axis=ax10)

    plot_acf_fit(data_hcorr_subs2,
                 params_hcorr_subs2.target_label,
                 title="TFH2",
                 target_axis=ax11)
    plt.tight_layout()
    plt.show()

def fit_gpr_gpkernel_subsets(return_mode="standard"):
    """
    For subsets, determine acf, fit acf with gp, use acf-gp as kernel function in gpr to fit timeseries

    :param return_mode: "If "standard", the signed returns are used. If "abs" absolute values of returns are used"
    :type return_mode: str

    :return: ---
    :rtype: None
    """
    params_lcorr_subs1 = Parameters(start_date="1993-01-01",
                                    end_date="1995-12-31",
                                    tick_interval_x=180,
                                    use_return=True,
                                    prediction_horizon=10,
                                    target_label="Adj Close",
                                    return_mode=return_mode,
                                    kernel_fct="gp_kernel",
                                    test_data_size=0.1,
                                    plot_line_tr_data=True)

    params_lcorr_subs2 = Parameters(start_date="2011-01-01",
                                    end_date="2012-12-31",
                                    tick_interval_x=180,
                                    use_return=True,
                                    prediction_horizon=10,
                                    target_label="Adj Close",
                                    return_mode=return_mode,
                                    kernel_fct="gp_kernel",
                                    test_data_size=0.1,
                                    plot_line_tr_data=True)

    params_hcorr_subs1 = Parameters(start_date="2000-01-01",
                                    end_date="2003-12-31",
                                    tick_interval_x=360,
                                    use_return=True,
                                    prediction_horizon=10,
                                    target_label="Adj Close",
                                    return_mode=return_mode,
                                    kernel_fct="gp_kernel",
                                    test_data_size=0.1,
                                    plot_line_tr_data=True)

    params_hcorr_subs2 = Parameters(start_date="2008-01-01",
                                    end_date="2009-12-31",
                                    tick_interval_x=180,
                                    use_return=True,
                                    prediction_horizon=10,
                                    target_label="Adj Close",
                                    return_mode=return_mode,
                                    kernel_fct="gp_kernel",
                                    test_data_size=0.1,
                                    plot_line_tr_data=True)

    # Parameters

    # Load dataset
    raw_data = load_msft(params_lcorr_subs1)
    params_lcorr_subs2.target_label = params_lcorr_subs1.target_label
    params_hcorr_subs1.target_label = params_lcorr_subs1.target_label
    params_hcorr_subs2.target_label = params_lcorr_subs1.target_label

    fit_gp(raw_data,
           params_lcorr_subs1,
           prediction_horizon_mode="day",
           subsample_timeframe=False,
           prediction_mode="all",
           prediction_horizon=360)

    fit_gp(raw_data,
           params_lcorr_subs2,
           prediction_horizon_mode="day",
           subsample_timeframe=False,
           prediction_mode="all",
           prediction_horizon=360)

    fit_gp(raw_data,
           params_hcorr_subs1,
           prediction_horizon_mode="day",
           subsample_timeframe=False,
           prediction_mode="all",
           prediction_horizon=360)

    fit_gp(raw_data,
           params_hcorr_subs2,
           prediction_horizon_mode="day",
           subsample_timeframe=False,
           prediction_mode="all",
           prediction_horizon=360)

def make_arg_parser():
    """
    Create ArgumentParser object

    :return: The created ArgumentParser object.
    :rtype: ArgumentParser
    """

    parser = ArgumentParser()
    parser.add_argument("--mode", type=str,
                        help="Mode: init_example/plot_return_ts/plot_return_full/plot_return_full_subs/plot_acf_subs/plot_gpr_gpkernel_subs",
                        required=False)
    parser.add_argument("--return_mode", type=str,
                        help="If not set or set to standard, returns are used. If set to abs, abs returns are used")
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

    if args.return_mode is None:
        return_mode = "standard"
    else:
        return_mode = args.return_mode

    if mode == "init_example":
        run_initial_example()
    elif mode == "plot_return_ts":
        plot_return_ts(return_mode=return_mode)
    elif mode == "plot_return_full":
        plot_return_full(return_mode=return_mode)
    elif mode == "plot_return_full_subs":
        plot_return_full_subs(return_mode=return_mode)
    elif mode == "plot_acf_subs":
        fit_acf_function_subsets(return_mode=return_mode)
    elif mode == "plot_gpr_gpkernel_subs":
        fit_gpr_gpkernel_subsets(return_mode=return_mode)
    else:
        print("Invalid mode")

    # fit_gp(raw_data, parameters, prediction_horizon_mode="day", subsample_timeframe=True, prediction_mode="all")
    # plot_sliding_window_autocorr(raw_data, "Return", "dt", window_size=180)


if __name__ == '__main__':
    #fit_gpr_gpkernel_subsets(return_mode="abs")
    #fit_acf_function_subsets(return_mode="abs")
    main()
