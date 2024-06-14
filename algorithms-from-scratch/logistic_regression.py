"""
Description:
Implementation of the Logistic Regression Model from scratch.

Objective:
For the author to understand the inner workings of the algorithm.

Algorithm Logic:
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

    def sigmoid(self, z):
        """
        Compute the sigmoid of z.

        Refer to sigmoid function script for detailed explaination.

        Args:
            z: A scalar or numpy array of any size.

        Returns
            s: sigmoid(z)
        """
        return 1 / (1 + np.exp(-z))

    def propogate(self, weights, bias, X, Y):
        pass
