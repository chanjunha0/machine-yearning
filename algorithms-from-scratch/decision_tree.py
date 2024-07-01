import numpy as np
from typing import List, Tuple


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        Initialise a node in the decision tree.

        Args:
            feature (int, optional): The index of the feature used for splitting at this node.
            threshold (float, optional): The threshold value for the split.
            left (Node, optional): The left child node.
            right (Node, optional): The right child node.
            value (Any, optional): The predicted value if this is a leaf node.

        If 'value' is set, this node is a leaf node.
        If 'feature' and 'threshold' are set, this is an internal node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """
        Check if the current node is a leaf node.

        Returns:
            bool: True if this is a leaf node, False otherwise

        A node is considered a leaf if it has a value assigned to it,
        which represents the prediction for samples that reach this node.
        """
        return self.value is not None


class DecisionTree:
    """
    This class implements a decision tree classifier from scratch.

    In a decision tree, we aim to split the dataset at each node into two subsets (left and right) such
    that the data in each subset is as homogeneous(more samples from the same class) as possible.

    This homogeneity is typically measured using criteria like Gini Impurity. However other methods like
    entrophy or information gain can also be used.
    """

    def __init__(self, max_depth=None):
        """
        Initializes the Decision Tree.

        Args:
            max_depth (int): The maximum depth of the tree. If None, the tree will grow until all leafs are pure (one class only).
        """
        self.max_depth = max_depth
        self.tree = None

    def gini(self, class0: List[int], class1: List[int]) -> float:
        """
        Calculates the Gini Impurity score for 2 lists of numbers representing the labels

        The Gini impurity quantifies the probability of making an incorrect classification based on the distribution of classes within a node.

        A node with a lower Gini impurity indicates a more homogeneous  set of data points (majority belonging to one class)

        Conversely, a higher Gini impurity indicates a more heterogeneous node.

        Math:
        For a given node n, the Gini impurity score is given as:

        i(n) = 1 - p0^2 - p1^2

        where
        p0 is the proportion of class 0 in that node.
        p1 is the proportion of class 1 in that node.

        Steps:
        1. Calculate the proportion of each class
        2. Implement the Gini Impurity Formula
        """
        # Calculate the total number of labels
        total_labels = len(class0) + len(class1)

        # Calculate the proportion of each class
        p0 = len(class0) / total_labels
        p1 = len(class1) / total_labels

        # Calculate the Gini Impurity Score
        gini_impurity = 1 - p0**2 - p1**2

        return gini_impurity

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fits the decision tree classifier to the training data.

        Args:
            X_train (np.ndarray): A m*n train array with m training examples of 'n' features.
            y_train (np.ndarray): A m*1 array of labels.
        """
        self.tree = self._build_tree(X_train, y_train)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """
        Recursively builds the decision tree.

        This function implements the core logic of the decision tree algorithm. It recursively
        splits the data based on the best feature and threshold, creating child nodes until
        a stopping criterion is met.

        Args:
            X (np.ndarray): A 2D array of shape (n_samples, n_features) containing the feature data.
            y (np.ndarray): A 1D array of shape (n_samples) containing the target labels.
            depth (int, optional): The current depth of the tree. Defaults to 0.

        Returns:
            Node: The root node of the decision tree (or subtree) built from the previous data.

        The function performs the following steps:
        1. Checks if a stopping criterion is met (max depth reached, pure node or too few samples)
        2. If a stopping criterion is met, creates a leaf node with the most common label.
        3. Otherwise, finds the best feature and threshold to split the data.
        4. Splits the data and recursively builds left and right subtrees.
        5. Returns a new internal node with the split information and child nodes.
        """
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # 1. Check stopping criteria
        if depth == self.max_depth or num_labels == 1 or num_samples < 2:
            # Majority voting
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # 2. Find the best split
        best_feature, best_threshold = self._best_split(X, y, num_samples, num_features)

        # 3. Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left_node = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_node = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)

        # 4. Create current node
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_node,
            right=right_node,
        )

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, num_rows: int, num_features: int
    ) -> Tuple[int, float]:
        """
        Finds the best feature and threshold to split the data based on Gini impurity at that node.

        Args:
            X (np.ndarray): A two-dimensional NumPy array of shape (num_rows, num_features) containg the features values of the dataset.
            y (np.ndarray): A one-dimensional NumPy array of shape (num_rows) containing the labels corresponding to the dataset.
            num_rows (int): The number of rows in the dataset.
            num_features (int): The number of features in the dataset.

        Returns:
            Tuple[int, float]: A tuple containing:
            - The index of the best feature to split on.
            - The threshold value for the best feature.

        """
        best_gini = float("inf")
        best_feature, best_threshold = None, None

        for feature in range(num_features):

            # Extract all unique values from the specified feature column
            thresholds = np.unique(X[:, feature])

            # Each threshold is a candidate for splitting the data into two groups
            for threshold in thresholds:
                left_idxs, right_idxs = self._split(X[:, feature], threshold)

                # Skip if threshold does not effectively split the data into 2 groups
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                # Extract the labels from y based on the indices
                y_left = y[left_idxs]
                y_right = y[right_idxs]

                # Calculate the Gini Impurity for each sub tree
                gini_left = self.gini(y_left == 0, y_left == 1)
                gini_right = self.gini(y_right == 0, y_right == 1)

                # Calculate the weighted gini of that node
                weighted_gini = (len(left_idxs) * gini_left) + (
                    len(right_idxs) * gini_right
                ) / num_rows

                # Update if a lower impurity value (more homogenous) is found
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _split(
        self, X_column: np.ndarray, split_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits the data based on the given feature and threshold.

        Args:
            X_column (np.ndarray): A one-dimensional NumPy array representing the feature column to split.
            split_threshold (float): The threshold value to split the data on.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the two NumPy arrays:
            - The first array contains the indices of the elements that are less than or equal to the split_threshold.
            - The second array contains the indices of the elements that are greater than the split_threshold.
        """
        # note that idx is a commonly used abbreviation for indices
        # Return index 0 since np.where returns a tuple of arrays
        left_idxs = np.where(X_column <= split_threshold)[0]
        right_idxs = np.where(X_column > split_threshold)[0]
        return left_idxs, right_idxs

    def _most_common_label(self, y: np.ndarray) -> int:
        """
        Determines the most common label in a given set of labels.

        Args:
            y (np.ndarray): A one-dimensional NumPy array of labels.

        Returns:
            int: The index of the first most common label in the array.
        """
        counts = np.bincount(y)
        return np.argmax(counts)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X (np.ndarray): 2D array of shape (n_samples, n_features)
        Returns:
            np.ndarray: Predicted class labels
        """
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        """
        Traverse the tree to make a prediction for a single sample.

        Args:
            x (np.ndarray): 1D array of shape (n_features,) representing a single sample
            node (Node): The current node in the decision tree

        Returns:
            int: Predicted class label
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
