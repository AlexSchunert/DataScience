from sklearn.metrics.pairwise import rbf_kernel as rbf_kernel_sklearn
from pandas import DataFrame
from numpy import ndarray
from gp_lib import GPPosterior, predict_gpr

# Todo: Remove "dt"
def rbf_kernel(input_left, input_right, length_scale=10.0, output_scale=15.0):
    """
    Calculates the values of the radial basis function kernel for input_left and input_right.
    Radial basis function is defined as rbf(a,b) = output_scale*exp(-((a-b)**2)/(2*length_scale**2))

    :param input_left: DataFrame containing input a. Must have column labeled "dt"
    :type input_left: DataFrame
    :param input_right: DataFrame containing input b. Must have column labeled "dt"
    :type input_right: DataFrame
    :param length_scale: Radial basis function length scale
    :type length_scale: float
    :param output_scale: Radial basis function output scale
    :type output_scale: float

    :return: Kernel matrix of inputs
    :rtype: ndarray
    """
    if input_left["dt"].values.ndim == 1:
        a = input_left["dt"].values.reshape(-1, 1)
    elif input_left["dt"].values.ndim == 2:
        a = input_left["dt"].values
    else:
        return

    if input_right["dt"].values.ndim == 1:
        b = input_right["dt"].values.reshape(-1, 1)
    elif input_right["dt"].values.ndim == 2:
        b = input_right["dt"].values
    else:
        return

    rbf_gamma = 1.0 / (2.0 * length_scale ** 2)
    k_ab = output_scale * rbf_kernel_sklearn(a, b, gamma=rbf_gamma)

    return k_ab

def gp_kernel(input_left, input_right, gp_posterior):
    """
    Calculates the values of kernel estimated from acf fit with gp

    :param input_left: DataFrame containing input a. Must have column labeled "dt"
    :type input_left: DataFrame
    :param input_right: DataFrame containing input b. Must have column labeled "dt"
    :type input_right: DataFrame
    :param gp_posterior: Struct containing at least representer weights and predictive covariance
    :type gp_posterior: GPPosterior

    :return: Kernel matrix of inputs
    :rtype: ndarray
    """
    if input_left["dt"].values.ndim == 1:
        a = input_left["dt"].values.reshape(-1, 1)
    elif input_left["dt"].values.ndim == 2:
        a = input_left["dt"].values
    else:
        return

    if input_right["dt"].values.ndim == 1:
        b = input_right["dt"].values.reshape(-1, 1)
    elif input_right["dt"].values.ndim == 2:
        b = input_right["dt"].values
    else:
        return




def compute_kernel_matrices(predict_data,
                            train_data,
                            kernel_type,
                            rbf_length_scale=None,
                            rbf_output_scale=None,
                            gp_posterior=None):
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
    :param kernel_type: Indicates type of kernel function used. Currently "rbf" and gp are supported
    :type kernel_type: str
    :param rbf_length_scale: Radial basis function length scale. Mandatory if kernel_type="rbf"
    :type rbf_length_scale: float
    :param rbf_output_scale: Radial basis function output scale. Mandatory if kernel_type="rbf"
    :type rbf_output_scale: float
    :param gp_posterior: Instance of class GPPosterior. Contains representer weights and pred. Cov.
    :type gp_posterior: GPPosterior

    :return: Tuple containing k_xx, k_zx, and k_zz
    :rtype: (ndarray,ndarray,ndarray)
    """

    if kernel_type == "rbf":
        if rbf_length_scale is None or rbf_output_scale is None:
            return
        else:
            k_xx = rbf_kernel(train_data, train_data, length_scale=rbf_length_scale, output_scale=rbf_output_scale)
            k_zx = rbf_kernel(predict_data, train_data, length_scale=rbf_length_scale, output_scale=rbf_output_scale)
            k_zz = rbf_kernel(predict_data, predict_data, length_scale=rbf_length_scale, output_scale=rbf_output_scale)
    else:
        return

    return k_xx, k_zx, k_zz