from sklearn.metrics.pairwise import rbf_kernel as rbf_kernel_sklearn
from pandas import DataFrame
from numpy import ndarray

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
