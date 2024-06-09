from numpy import linalg, eye, sqrt, diagonal
import pandas as pd

from utils import compute_kernel_matrices, construct_prediction_result


def condition_gpr(train_data, k_xx, col_id, sigma_data):
    """
    Description of function
    
    :param param1: Description of param1
    :type param1: Type of param1
    
    :return: Description of return value
    :rtype: Type of return value
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
    ## prediction
    price_predicted = k_zx @ alpha
    ## stdev of prediction
    # + sigma_price ** 2 * np.eye(k_zz.shape[0])
    cov_prediction = k_zz - k_zx @ predictive_cov @ k_zx.T
    std_prediction = sqrt(diagonal(cov_prediction))

    return price_predicted, std_prediction

def gp_process(prediction_data,
               train_data,
               target_quantity_idx,
               result_label,
               sigma_measurement,
               rbf_length_scale,
               rbf_output_scale):
    # Fit gp
    k_xx, k_zx, k_zz = compute_kernel_matrices(prediction_data,
                                               train_data,
                                               rbf_length_scale,
                                               rbf_output_scale,
                                               "rbf")
    # condition on data
    alpha, predictive_cov = condition_gpr(train_data, k_xx, target_quantity_idx, sigma_measurement)
    # predict for all data
    mu_predicted, std_prediction = predict_gpr(alpha, k_zx, k_zz, predictive_cov)
    # create result
    result = construct_prediction_result(prediction_data, mu_predicted, std_prediction,
                                         result_label=result_label)

    return result

