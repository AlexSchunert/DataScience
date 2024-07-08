from sklearn.metrics.pairwise import rbf_kernel as rbf_kernel_sklearn
from pandas import DataFrame
from numpy import ndarray, meshgrid, abs, unique, where, squeeze
from gp_lib import GPPosterior, predict_gpr
from utils import create_index_matrix

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

    if gp_posterior.kernel_fct[0] == "rbf":

        rbf_length_scale = gp_posterior.kernel_fct[1]
        rbf_output_scale = gp_posterior.kernel_fct[2]

        # Calculate possible lags in data
        A, B = meshgrid(a, b)
        abs_diff_mat = abs(A - B)
        abs_diff_unique_vec = unique(abs_diff_mat.reshape(-1))
        x_test = DataFrame({
            "dt": abs_diff_unique_vec
        })

        k_zx = rbf_kernel(x_test, gp_posterior.x_training, length_scale=rbf_length_scale,
                          output_scale=rbf_output_scale)
        k_zz = rbf_kernel(x_test, x_test, length_scale=rbf_length_scale, output_scale=rbf_output_scale)
        kernel_matrix_values, _ = predict_gpr(gp_posterior.repr_weights,
                                              k_zx,
                                              k_zz,
                                              gp_posterior.predictive_cov)

        # Set covariances for time differences that are not in the data to zero
        #kernel_matrix_values[where(x_test["dt"] > gp_posterior.x_training["dt"].max())[0]] = 0

        # Construct kernel matrix from kernel_matrix_values and x_test
        idx_mat = create_index_matrix(abs_diff_mat, abs_diff_unique_vec)
        k_ab = squeeze(kernel_matrix_values[idx_mat])

    else:
        k_ab = None

    # Transpose for some reason
    return k_ab.T


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
    elif kernel_type == "gp_kernel":
        if gp_posterior is None:
            return
        else:
            k_xx = gp_kernel(train_data, train_data, gp_posterior)
            k_zx = gp_kernel(predict_data, train_data, gp_posterior)
            k_zz = gp_kernel(predict_data, predict_data, gp_posterior)
    else:
        return

    return k_xx, k_zx, k_zz
