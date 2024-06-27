"""
Description:
Implementation of the Logistic Regression Model from scratch.

Objective:
For the author to understand the inner workings of the algorithm.

Algorithm Logic:
1.

Reference Materials:
https://www.analyticsvidhya.com/blog/2022/02/implementing-logistic-regression-from-scratch-using-python/
https://www.youtube.com/watch?v=YYEJ_GUguHw
"""

import numpy as np
from typing import Tuple
from sigmoid_function import sigmoid
from standardisation import standardize_column


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    """
    1. Initialize weights as zero
    2. Initialize bias as zero
    3. For each data point
    4. Predict probability using sigmoid function
    5. Calculate error
    6. Use gradient descent to figure out new weight and bias
    7. Repeat n times
    """

    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeroes(self.features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_predictions = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_predictions)

    def predict():
        pass
