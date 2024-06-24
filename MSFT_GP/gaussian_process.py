from numpy import linalg, eye, sqrt, diagonal, ndarray, empty
import pandas as pd
from dataclasses import dataclass
from utils import construct_prediction_result
from kernel_functions import rbf_kernel


@dataclass
class GPPosterior:
    """
    Contains the posterior of a gp given by represented weights and predictive covariance
    """

    repr_weights: ndarray = empty,
    predictive_cov: ndarray = empty

    def __int__(self,
                repr_weights,
                predictive_cov):
        """
        Inits GPPosterior

        :param repr_weights: Representer weights determined during conditioning
        :type repr_weights: ndarray
        :param predictive_cov: Predictive covariance determined during conditioning
        :type predictive_cov: ndarray

        :return: ---
        :rtype: None
        """
        self.repr_weights = repr_weights
        self.predictive_cov = predictive_cov


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


def condition_gpr(train_data, k_xx, col_id, sigma_data):
    """
    Conditions gaussian process using data => calulate vector of representer weights and the
    predictive covariance matrix

    :param train_data: Data used for conditioning. Must contain data with label col_id
    :type train_data: pd.DataFrame
    :param k_xx: Kernel matrix of training data
    :type k_xx: ndarray
    :param col_id: Identifier of target quantity used for conditioning
    :type col_id: str
    :param sigma_data: Standard deviation of the quantity fused for conditioning
    :type sigma_data: float
    
    :return: tuple(representer weights, predictive covariance)
    :rtype: (ndarray,ndarray)
    """
    # Compute kernel for data points => K_xx

    # Construct predictive covariance
    ## Compute Cholesky decomposition
    L = linalg.cholesky(k_xx + sigma_data ** 2 * eye(k_xx.shape[0]))
    ## Compute inverse => predictive covariance
    predictive_cov = linalg.inv(L.T).dot(linalg.inv(L))

    # Compute representer weights => zero prior
    alpha = predictive_cov @ train_data[col_id].values.reshape(-1, 1)

    return alpha, predictive_cov


def predict_gpr(alpha, k_zx, k_zz, predictive_cov):
    """
    Predict gp mean and standard-deviation for training data z using the kernel matrix of the prediction data k_zz
    and the cross-kernel matrix of prediction data and training data k_zx.

    :param alpha: Vector of representer weights
    :type alpha: ndarray
    :param k_zx: Cross-kernel matrix of prediction data and training data
    :type k_zx: ndarray
    :param k_zz: Kernel matrix of prediction data
    :type k_zz: ndarray
    :param predictive_cov:
    :type predictive_cov: ndarray

    :return: tuple(mean,std)
    :rtype: (ndarray,ndarray)
    """

    ## prediction
    mean_prediction = k_zx @ alpha
    ## stdev of prediction
    # + sigma_price ** 2 * np.eye(k_zz.shape[0])
    cov_prediction = k_zz - k_zx @ predictive_cov @ k_zx.T
    std_prediction = sqrt(diagonal(cov_prediction))

    return mean_prediction, std_prediction


# Todo: Add optional input time_index to generalize "dt"
# Todo: Generalize kernel function
def gp_process(test_data,
               train_data,
               target_quantity_idx,
               result_label,
               sigma_measurement,
               rbf_length_scale,
               rbf_output_scale):
    """
    Executes:
        - Computation of kernel matrices
        - Conditioning of gp to training data
        - Prediction of gp mean and std for test data
        - Construction of result


    :param test_data: Data used for prediction => Must contain field "dt"
    :type test_data: pd.DataFrame
    :param train_data: Data used for training (conditioning) => Must contain field "dt" and target_quantity_idx
    :type train_data: pd.DataFrame
    :param target_quantity_idx: String-label of the target quantity to be used
    :type target_quantity_idx: str
    :param result_label: String-label of the resulting mean values
    :type result_label: str
    :param sigma_measurement: Standard deviation of the target quantity
    :type sigma_measurement: float
    :param rbf_length_scale: Length scale of radial basis function
    :type rbf_length_scale: float
    :param rbf_output_scale: Output Scale of radial basis function
    :type rbf_output_scale: float

    :return: DataFrame containing the columns: Date, dt, result_label, and std.
    :rtype: DataFrame
    """

    # Fit gp
    k_xx, k_zx, k_zz = compute_kernel_matrices(test_data,
                                               train_data,
                                               "rbf",
                                               rbf_length_scale=rbf_length_scale,
                                               rbf_output_scale=rbf_output_scale)
    # condition on data
    alpha, predictive_cov = condition_gpr(train_data, k_xx, target_quantity_idx, sigma_measurement)
    # predict for all data
    mu_predicted, std_prediction = predict_gpr(alpha, k_zx, k_zz, predictive_cov)
    # create result
    result = construct_prediction_result(test_data, mu_predicted, std_prediction,
                                         result_label=result_label)

    gp_posterior = GPPosterior(alpha, predictive_cov)

    return result, gp_posterior
