"""
The following script below is the implementation of the Logistic Regression Model from scratch.

The purpose is for the author to understand the inner workings of the algorithm.

Overall Logic:
1.
"""

import numpy as np
from typing import Tuple


class LogisticRegression:
    def initialize_parameters(dim: int) -> Tuple[np.ndarray, float]:
        """
        Initializes the parameters (weight and bias) for logistic regression.

        Args:
            dim (int): The number of features in the input data.

        Returns:
            weights (numpy.ndarray): A column vector of shape (dim,1) initialized to zeros.
            bias (float): The bias term initialized to zero. (Y Intercept)
        """
        weights = np.zeros((dim, 1))
        bias = 0
        return weights, bias

    def sigmoid(z):
        """
        Refer to sigmoid function script for explaination.
        """
        return 1 / (1 + np.exp(-z))

    def propogate(w, b, X, Y):
        pass
