import pandas as pd
from utils import construct_prediction_result
from kernel_functions import compute_kernel_matrices
from gp_lib import condition_gpr, predict_gpr, GPPosterior


# Todo: Add optional input time_index to generalize "dt"
# Todo: Generalize kernel function
def gp_process(test_data,
               train_data,
               target_quantity_idx,
               result_label,
               sigma_measurement,
               rbf_length_scale=None,
               rbf_output_scale=None,
               gp_posterior=None,
               kernel_fct="rbf"):
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
    :param gp_posterior: Instance of class GPPosterior. Contains representer weights and pred. Cov.
    :type gp_posterior: GPPosterior
    :param kernel_fct: Type of kernel function to be used. Currently, either "rbf" or "gp_kernel"
    :type kernel_fct: str


    :return: DataFrame containing the columns: Date, dt, result_label, and std.
    :rtype: DataFrame
    """

    # Fit gp
    k_xx, k_zx, k_zz = compute_kernel_matrices(test_data,
                                               train_data,
                                               kernel_fct,
                                               rbf_length_scale=rbf_length_scale,
                                               rbf_output_scale=rbf_output_scale,
                                               gp_posterior=gp_posterior)
    # condition on data
    alpha, predictive_cov = condition_gpr(train_data, k_xx, target_quantity_idx, sigma_measurement)
    # predict for all data
    mu_predicted, std_prediction = predict_gpr(alpha, k_zx, k_zz, predictive_cov)
    # create result
    result = construct_prediction_result(test_data, mu_predicted, std_prediction,
                                         result_label=result_label)

    if kernel_fct == "rbf":
        kernel_fct = ["rbf", rbf_length_scale, rbf_output_scale]
    elif kernel_fct == "gp_kernel":
        kernel_fct = ["gp_kernel", 0, 0]
    else:
        print("Warning: Could not set kernel function in gp_posterior")

    x_training = train_data.loc[:, ["dt"]]
    gp_posterior = GPPosterior(alpha, predictive_cov, x_training, kernel_fct)

    return result, gp_posterior
