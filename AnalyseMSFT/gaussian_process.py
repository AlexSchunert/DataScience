from numpy import linalg, eye, sqrt, diagonal
import pandas as pd


def condition_gpr(train_data, k_xx, col_id, sigma_price):
    # Compute kernel for data points => K_xx

    # Construct predictive covariance
    ## Compute Cholesky decomposition
    L = linalg.cholesky(k_xx + sigma_price ** 2 * eye(k_xx.shape[0]))
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
