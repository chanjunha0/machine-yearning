import numpy as np
from typing import List


class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000) -> None:
        """
        Initialise the Logistic Regression model.

        Args:
            learning_rate (float): The learning rate for gradient descent. Defaults to 0.01.
            num_iterations (int): The number of iterations for training. Defaults to 1000.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid function.

        Args:
            z (np.ndarray): Input array

        Returns:
            np.ndarray: Sigmoid of the input array.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the logistic regression model to the training data.

        Args:
            X (np.ndarray): Training input samples.
            y (np.ndarray): Target values.
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> List[int]:
        """
        Predict class labels for samples in X.

        Args:
            X (np.ndarray): Samples to predict.

        Returns:
            List[int]: Predicted class label for each sample.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]
