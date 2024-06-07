from sklearn.metrics.pairwise import rbf_kernel as rbf_kernel_sklearn


def rbf_kernel(input_left, input_right, length_scale=10.0, output_scale=15.0):
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
