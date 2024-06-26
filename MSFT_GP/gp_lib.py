from numpy import linalg, eye, sqrt, diagonal, ndarray, empty
from dataclasses import dataclass

@dataclass
class GPPosterior:
    """
    Contains the posterior of a gp given by represented weights and predictive covariance
    """

    repr_weights: ndarray = empty,
    predictive_cov: ndarray = empty,
    x_training: ndarray = empty,
    kernel_fct: list = [],

    def __int__(self,
                repr_weights,
                predictive_cov,
                x_training,
                kernel_fct):
        """
        Inits GPPosterior

        :param repr_weights: Representer weights determined during conditioning
        :type repr_weights: ndarray
        :param predictive_cov: Predictive covariance determined during conditioning
        :type predictive_cov: ndarray
        :param x_training: X-Values of training data. Necessary to calculate kernel matrices
        :type x_training: ndarray
        :param kernel_fct: Contains all necessary information about used kernel function.
        :type kernel_fct: list

        :return: Created GPPosterior
        :rtype: GPPosterior
        """
        self.repr_weights = repr_weights
        self.predictive_cov = predictive_cov
        self.kernel_fct = kernel_fct

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
