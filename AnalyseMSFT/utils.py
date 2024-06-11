import pandas as pd
from sklearn.model_selection import train_test_split as train_test_split_sklearn
from dataclasses import dataclass
from numpy import ndarray
from kernel_functions import rbf_kernel


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

    def __init__(self,
                 test_data_size=0.5,
                 rbf_length_scale=5.0,  # 20.0
                 rbf_output_scale=20.0,
                 sigma_price=0.1,
                 sigma_return=1.0,
                 sigma_used=0.0,
                 target_label="High",
                 use_return=True,
                 plot_shading_mode="2-sigma",
                 start_date="2000-01-01",
                 end_date="2006-12-31",
                 prediction_horizon=10,
                 num_iter_error_stat=1000,
                 num_data_points_gp_fit=10,
                 histogram_num_bins=100):
        """
        Initialises the parameter dataclass

        :param test_data_size: Portion of test data in case subsampling of input time frame is enabled
        :type test_data_size: float
        :param rbf_length_scale: Length scale of radial basis function
        :type rbf_length_scale: float
        :param rbf_output_scale: Output Scale of radial basis function
        :type rbf_output_scale: float
        :param sigma_price: Standard deviation of price
        :type sigma_price: float
        :param sigma_return: Standard deviation of return
        :type sigma_return: float
        :param sigma_used: Either set to sigma_price or sigma_return depending on use_return
        :type sigma_used: float
        :param target_label: String label of target quantity.
        :type target_label: str
        :param use_return: If true, return is computed for target quantity and used for further computation
        :type use_return: bool
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
        self.plot_shading_mode = plot_shading_mode
        self.start_date = start_date
        self.end_date = end_date
        self.prediction_horizon = prediction_horizon
        self.num_iter_error_stat = num_iter_error_stat
        self.num_data_points_gp_fit = num_data_points_gp_fit
        self.histogram_num_bins = histogram_num_bins


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


def compute_kernel_matrices(predict_data, train_data, rbf_length_scale, rbf_output_scale, kernel_type="rbf"):
    """
    Computes all kernel matrices necessary for gp prediction:
    - k_xx: Kernel matrix of training data
    - k_zz: Kernel matrix of prediction data
    - k_zx: Cross kernel matrix of prediction and training data

    :param predict_data: Data used for prediction. Necessary for construction of k_zx and k_zz. Must contain
                         column labeled "dt"
    :type predict_data: pd.DataFrame
    :param train_data: Data used for training. Necessary for construction of k_zx and k_xx. Must contain
                       column labeled "dt"
    :type train_data: pd.DataFrame
    :param rbf_length_scale: Radial basis function length scale
    :type rbf_length_scale: float
    :param rbf_output_scale: Radial basis function output scale
    :type rbf_output_scale: float
    :param kernel_type: Indicates type of kernel function used. Currently only "rbf" is supported
    :type kernel_type: str

    :return: Tuple containing k_xx, k_zx, and k_zz
    :rtype: (ndarray,ndarray,ndarray)
    """

    if kernel_type == "rbf":
        k_xx = rbf_kernel(train_data, train_data, length_scale=rbf_length_scale, output_scale=rbf_output_scale)
        k_zx = rbf_kernel(predict_data, train_data, length_scale=rbf_length_scale, output_scale=rbf_output_scale)
        k_zz = rbf_kernel(predict_data, predict_data, length_scale=rbf_length_scale, output_scale=rbf_output_scale)
    else:
        return

    return k_xx, k_zx, k_zz


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
    result = pd.DataFrame({
        "Date": data["Date"].reset_index(drop=True),
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
