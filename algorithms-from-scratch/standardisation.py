import numpy as np


def standardize_column(column):
    """
    Standardizes the column of a given dataset.

    Args:
        column (numpy.ndarray): The column to be standardized.

    Returns:
        numpy.ndarray: The standardized column.

    Explanation:
        Standardization transforms the data to have a mean of 0 and a standard deviation of 1.
        This is useful for algorithms that assume or perform better when the input data is standardized.

        Mathematically:
            standardized_value = (original_value - mean) / standard_deviation

        Where:
            - 'mean' is the average of the column.
            - 'standard_deviation' measures the amount of variation or dispersion of the column values.
            - The output is a standardized column where each value represents the number of standard deviations away from the mean.

    Example:
        >>> import numpy as np
        >>> data = np.array([[1, 2, 3],
        ...                  [4, 5, 6],
        ...                  [7, 8, 9]])
        >>> column = data[:, 0]
        >>> standardized_column = standardize_column(column)
        >>> standardized_column
        array([-1.22474487,  0.        ,  1.22474487])

    References:
        - https://www.analyticsvidhya.com/blog/2022/02/implementing-logistic-regression-from-scratch-using-python/

    Additional Notes:
        - Standardization is often a preprocessing step for machine learning algorithms.
    """
    feature_mean = np.mean(column)
    feature_std = np.std(column)
    normalized_column = (column - feature_mean) / feature_std
    return normalized_column
