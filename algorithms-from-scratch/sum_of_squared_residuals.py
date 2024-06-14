import numpy as np


def sum_of_squared_residuals(y_true, y_pred):
    """
    Calculate the sum of the squared residuals (SSR) between the true and predicted values.

    The sum of squared residuals is a measure of the discrepancy between the observed data and the values predicted by a model.
    It is used in regression analysis to assess the goodness of fit of a model. A smaller SSR indicates a better fit.

    Args:
        y_true (array): True values
        y_pred (array): Predicted values

    Returns:
        float: The sum of the squared residuals.

    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y_pred = np.array([0.8, 2.1, 2.9, 4.2, 4.8])
        >>> ssr = sum_of_squared_residuals(y_true, y_pred)
        >>> print("Sum of Squared Residuals:", ssr)
        Sum of Squared Residuals: 0.14

    Notes:
        The sum of squared residuals is computed as follows:
        1. Calculate the residuals: residuals = y_true - y_pred
        2. Square each residual: squared_residuals = residuals ** 2
        3. Sum all the squared residuals: ssr = np.sum(squared_residuals)
    """
    residuals = y_true - y_pred
    squared_residuals = residuals**2
    ssr = np.sum(squared_residuals)
    return ssr
