import pandas as pd
from sklearn.model_selection import train_test_split as train_test_split_sklearn
from dataclasses import dataclass
from numpy import ndarray, zeros, abs, exp as np_exp, sin, empty, zeros_like, allclose, all
from numpy.linalg import eigh
from scipy.signal import correlate


@dataclass
class Parameters:
    """
    Dataclass for general parameters
    """

    test_data_size: float
    rbf_length_scale: float
    rbf_output_scale: float
    sigma_price: float  # Make sigma_price a function of time?
    sigma_return: float
    sigma_used: float
    target_label: str
    use_return: bool
    plot_shading_mode: str
    start_date: str
    end_date: str
    prediction_horizon: int
    num_iter_error_stat: int
    num_data_points_gp_fit: int
    histogram_num_bins: int
    kernel_fct: str
    plot_line_tr_data: bool
    plot_line_test_data: bool

    def __init__(self,
                 test_data_size=0.5,
                 rbf_length_scale=5.0,  # 20.0
                 rbf_output_scale=20.0,
                 sigma_price=0.1,
                 sigma_return=1.0,
                 sigma_used=0.0,
                 target_label="High",
                 use_return=True,
                 return_mode="Standard",
                 plot_shading_mode="2-sigma",
                 start_date="2000-01-01",
                 end_date="2006-12-31",
                 prediction_horizon=10,
                 num_iter_error_stat=1000,
                 num_data_points_gp_fit=10,
                 histogram_num_bins=100,
                 tick_interval_x=10,
                 kernel_fct="rbf",
                 plot_line_tr_data=False,
                 plot_line_test_data=False):
        """
        Initialises the parameter dataclass

        :param test_data_size: Portion of test data in case subsampling of input time frame is enabled
        :type test_data_size: float
        :param rbf_length_scale: Length scale of radial basis function. Not used if kernel_fct!="rbf"
        :type rbf_length_scale: float
        :param rbf_output_scale: Output Scale of radial basis function. Not used if kernel_fct!="rbf"
        :type rbf_output_scale: float
        :param sigma_price: Standard deviation of price. Not used if kernel_fct=="gp_kernel"
        :type sigma_price: float
        :param sigma_return: Standard deviation of return. Not used if kernel_fct=="gp_kernel"
        :type sigma_return: float
        :param sigma_used: Either set to sigma_price or sigma_return depending on use_return. Not used if kernel_fct=="gp_kernel"
        :type sigma_used: float
        :param target_label: String label of target quantity.
        :type target_label: str
        :param use_return: If true, return is computed for target quantity and used for further computation
        :type use_return: bool
        :param return_mode: "standard" for signed returns, "abs" for absolute values
        :type return_mode: str
        :param plot_shading_mode: Determines which region shall be shaded in plot_prediction_result. Currently only
                                  2-sigma is supported => region of 2 x standard deviation around mean
        :type plot_shading_mode: str
        :param start_date: Start date for considered timeframe in format YYYY-MM-DD.
        :type start_date: str
        :param end_date: End date for considered timeframe in format YYYY-MM-DD.
        :type end_date: str
        :param prediction_horizon: Days or number of data-points after end_date for which prediction is done.
                                   Note that only dates that are contained in the given data are considered.
                                   In case prediction_horizon specifies days, the data is searched for data
                                   points with dates in [end_date+1day, end_date+prediction_horizon].
                                   In case prediction_horizon specifies index, the data is searched for data
                                   points with index [end_idx+1, end_idx + prediction_horizon -1].
        :type prediction_horizon: int
        :param num_iter_error_stat: Number of iterations in gp_prediction_vs_martingale
        :type num_iter_error_stat: int
        :param num_data_points_gp_fit:  Number of datapoints used for conditioning in gp_prediction_vs_martingale
        :type num_data_points_gp_fit: int
        :param histogram_num_bins: Number of bins in histogram plot
        :type histogram_num_bins: int
        :param tick_interval_x: Tick on x-axis every tick_interval_x days
        :type tick_interval_x: int
        :param kernel_fct: Type of kernel function to be used. Currently, either "rbf" or "gp_kernel"
        :type kernel_fct: str
        :param plot_line_tr_data: If true, plot training data as line in plot_prediction_result
        :type plot_line_tr_data: bool
        :param plot_line_test_data: If true, plot test data as line in plot_prediction_result
        :type plot_line_test_data: bool

        :return: The created parameters dataclass
        :rtype: Parameters
        """
        self.test_data_size = test_data_size
        self.rbf_length_scale = rbf_length_scale
        self.rbf_output_scale = rbf_output_scale
        self.sigma_price = sigma_price
        self.sigma_return = sigma_return
        self.sigma_used = sigma_used
        self.target_label = target_label
        self.use_return = use_return
        self.return_mode = return_mode
        self.plot_shading_mode = plot_shading_mode
        self.start_date = start_date
        self.end_date = end_date
        self.prediction_horizon = prediction_horizon
        self.num_iter_error_stat = num_iter_error_stat
        self.num_data_points_gp_fit = num_data_points_gp_fit
        self.histogram_num_bins = histogram_num_bins
        self.tick_interval_x = tick_interval_x
        self.kernel_fct = kernel_fct
        self.plot_line_tr_data = plot_line_tr_data
        self.plot_line_test_data = plot_line_test_data


def load_msft(parameters):
    """
    Loads msft dataset from disk

    :param parameters: Contains general settings.
    :type parameters: Parameters

    :return: DataFrame containing msft data
    :rtype: pd.DataFrame
    """

    # Load data
    raw_data = pd.read_csv("../DataSets/MSFT.csv")
    # Extract timestamps
    time = pd.to_datetime(raw_data["Date"], format="%Y-%m-%d")
    # Convert timestamps to days from start
    delta_time = (time - time[0]).dt.days
    # Add column delta_time to raw_data
    raw_data["dt"] = delta_time

    # Compute 1-d return if necessary
    if parameters.use_return:
        raw_data = compute_return(raw_data, parameters.target_label)
        raw_data = raw_data.reset_index(drop=True)

        if parameters.return_mode == "abs":
            raw_data["AbsReturn"] = abs(raw_data["Return"])
            parameters.target_label = "AbsReturn"
            parameters.sigma_used = parameters.sigma_return
        else:

            parameters.target_label = "Return"
            parameters.sigma_used = parameters.sigma_return

    else:
        parameters.sigma_used = parameters.sigma_price

    return raw_data


def select_data_time(data_full, start_date, end_date, date_idx="Date"):
    """
    Select data from specified timeframe.

    :param data_full: Complete dataset. Must contain column labeled date_idx
    :type data_full: pd.DataFrame
    :param start_date: Start date of timeframe in format YYYY-MM-DD. Note that this day is included in the subset.
    :type start_date: str
    :param end_date: End date of timeframe in format YYYY-MM-DD. Note that this day is included in the subset.
    :type end_date:str
    :param date_idx: Label of date column. Column must contain dates as strings in format YYYY-MM-DD
    :type date_idx: str

    :return: DataFrame containing the data found for the specified timeframe
    :rtype: pd.DataFrame
    """
    idx = (data_full[date_idx] >= start_date) & (data_full[date_idx] <= end_date)
    data_subset = data_full[idx]
    return data_subset


def train_test_split(data, test_data_size):
    """
    Splits data into training- and test-data.

    :param data: Data to be split into training and test-data
    :type data: pd.DataFrame
    :param test_data_size: Portion of test data (e.g. 0.5 indicates an even split between test and training-data)
    :type test_data_size: float

    :return: Tuple containing trining- and test-data
    :rtype: tuple(pd.DataFrame,pd.DataFrame)
    """
    train_data, test_data = train_test_split_sklearn(data, test_size=test_data_size, random_state=42)
    train_data = train_data.sort_index()
    test_data = test_data.sort_index()

    return train_data, test_data


def construct_prediction_result(data, mu_predicted, std_predicted, result_label="prediction"):
    """
    Constructs result DataFrame from predicted gp mean and standard deviation.

    :param data: DataFrame with the same number of rows than mu_predicted and std_predicted. Must contain columns
                 with labels "Date" and "dt.". Used to set "Date" and "dt" columns of result
    :type data: pd.DataFrame
    :param mu_predicted: GP mean values predicted for data points in data
    :type mu_predicted: ndarray
    :param std_predicted: GP standard deviation predicted for data points in data
    :type std_predicted: ndarray
    :param result_label: Columns label for mu_predicted in result
    :type result_label: str

    :return: DataFrame containing the columns: "Date", "dt", parameters.result_label, and "std".
    :rtype: pd.DateFrame
    """
    # Create result series
    if "Date" in data.columns:
        result = pd.DataFrame({
            "Date": data["Date"].reset_index(drop=True),
            "dt": data["dt"].reset_index(drop=True),
            result_label: pd.Series(mu_predicted.reshape(-1)),
            "std": pd.Series(std_predicted)
        })
    else:
        result = pd.DataFrame({
            "dt": data["dt"].reset_index(drop=True),
            result_label: pd.Series(mu_predicted.reshape(-1)),
            "std": pd.Series(std_predicted)
        })

    return result


def compute_return(data, target_price_idx):
    """
    Computes 1-day return for column identified by label target_price_idx.

    :param data: Stock data
    :type data: pd.DataFrame
    :param target_price_idx: Label of column for which 1-day return shall be calculated
    :type target_price_idx: str

    :return: DataFrame with added column "Return"
    :rtype: pd.DataFrame
    """
    data["Return"] = 100.0 * data[target_price_idx].diff() / data[target_price_idx].shift(1)
    return data.dropna(subset=["Return"])


def autocorrelations_sliding_window(time_series, window_size):
    """
    Generated by ChatGPT. Basically slides a window of length window_length over the timeseries and calculates the
    autocorrelation function. The resulting autocorrelation functions are stored in a grid.

    :param time_series: Autocorrelation is calculated for this signal
    :type time_series: ndarray
    :param window_size: Length of autocorrelation window.
    :type window_size: ndarray

    :return: matrix of autocorrelations
    :rtype: ndarray
    """
    n = len(time_series)
    autocorrelations = zeros((n - window_size + 1, window_size))

    for i in range(n - window_size + 1):
        window = time_series[i:i + window_size]
        autocorr = correlate(window, window, mode='full') / window_size

        # Extract the relevant part of the autocorrelation (positive lags)
        autocorr = autocorr[window_size - 1:window_size + window_size - 1]
        autocorr = autocorr / autocorr[0]
        autocorrelations[i, :] = autocorr

    return autocorrelations


def create_index_matrix(M, v):
    """
    Returns an index matrix Idx for a matrix M and a vector v such that v(I_ij)==M_ij. Note that v must contain all
    elements in M exactly once
    
    :param M: Matrix M
    :type M: ndarray
    :param v: Vector v
    :type v: ndarray

    :return: Index matrix Idx
    :rtype: ndarray
    """

    # Create a dictionary that maps each value in v to its index
    value_to_index = {value: idx for idx, value in enumerate(v)}

    # Initialize the index matrix I with the same shape as M
    Idx = zeros_like(M, dtype=int)

    # Fill the index matrix with the corresponding indices from v
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Idx[i, j] = value_to_index[M[i, j]]

    return Idx


def is_positive_semidefinite(matrix):
    """
    Checks if input matrix is positive semi-definite

    :param matrix: Matrix to be checked
    :type matrix: ndarray

    :return: True if matrix is positive-semi-definite, false otherwise
    :rtype: bool
    """
    # Check if matrix is square
    rows, cols = matrix.shape
    if rows != cols:
        return False

    # Check if matrix is symmetric
    if not allclose(matrix, matrix.T):
        return False

    # Compute eigenvalues
    eigenvalues, _ = eigh(matrix)

    # Check if all eigenvalues are non-negative
    return all(eigenvalues >= 0)
