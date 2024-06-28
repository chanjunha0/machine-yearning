import numpy as np
from typing import Tuple

from .decision_tree import DecisionTree


class RandomForest:
    """
    A simple implementation of a Random Forest Classifier.

    This class creates an ensemble of decision trees, each trained on a
    bootstrap sample of the input data. Predictions are made by majority
    voting among all trees.

    Attributes:
        n_trees (int): The number of trees in the forest.
        max_depth (int): The maximum depth allowed for each tree.
        trees (list): A list to store the decision trees.
    """

    def __init__(self, n_trees: int = 5, max_depth: int = None):
        """
        Initialize the RandomForest

        Args:
            n_trees (int): The number of trees to create in the forest. Default is 5.
            max_depth (int): Maximum depth for each tree. If none, the trees will grow
            till a leaf node.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def bootstrap_sample(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a bootstrap sample of the input data.

        This method samples data points from the original dataset with replacement,
        meaning the same data point can be selected multiple times.
        This results in a new dataset of the same size as the original dataset
        but with some data points repeated and some omitted.

        The returned sample will be used to train one decision tree in
        the fit() method.

        Args:
            X (numpy.ndarray): The input features
            y (numpy.ndarray): The target values

        Returns:
            tuple: A tuple containing the bootstrap sample of X and y.

        """
        # Extract the number of samples
        n_samples = X.shape[0]
        # Randomly select n_samples from range (0, n_samples)
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the random forest to the training data.

        This method creates multiple decision trees, each trained on a
        bootstrap sample of the input data.

        `RandomForest` instance will have a list of trained decision trees
        stored in the `self.trees` attribute to be used for predictions.

        Args:
            X (numpy.ndarray): The input values for training.
            y (numpy.ndarray): The target values for training

        Example:
        >>>> self.trees = [
            <DecisionTree object at memory_location_1>,
            <DecisionTree object at memory_location_2>,
            ...
            <DecisionTree object at memory_location_(n_trees)>
        ]
        """
        # Used to store the decision trees
        self.trees = []

        for _ in range(self.n_trees):

            # Create a decision tree instance with specified max depth
            tree = DecisionTree(max_depth=self.max_depth)

            # Create a bootstrapped sample of the dataset.
            X_sample, y_sample = self.bootstrap_sample(X, y)

            # Trains the decision tree on the boostrapped sample.
            tree.fit(X_sample, y_sample)

            # Append the decision tree to the list.
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        This method makes predictions using all trees in the forest and then uses majority
        voting to determine the final prediction for each sample.

        Args:
            x (numpy.ndarray): The input features to make predictions on.

        Returns:
            numpy.ndarray: The predicted class labels.
        """
        # Collect predictions from each tree into an array
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        # Swap axes of the array
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        # For each prediction, count the occurences of each class label and return the max
        y_pred = np.array([np.bincount(tree_pred).argmax() for tree_pred in tree_preds])
        return y_pred
