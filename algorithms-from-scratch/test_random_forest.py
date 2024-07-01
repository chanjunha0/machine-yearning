import pytest
import numpy as np
from numpy.typing import NDArray

from .decision_tree import DecisionTree
from .random_forest import RandomForest

# Constants
X_train: NDArray[np.int_] = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y_train: NDArray[np.int_] = np.array([0, 0, 1, 1, 1])
X_test: NDArray[np.int_] = np.array([[2, 3], [6, 7]])


@pytest.fixture
def random_forest() -> RandomForest:
    """
    Fixture to create a RandomForest instance.

    Returns:
        RandomForest: An instance of the RandomForest class with 5 trees and max depth of 3.
    """
    return RandomForest(n_trees=5, max_depth=3)


def test_bootstrap_sample(random_forest: RandomForest) -> None:
    """
    Test the bootstrap_sample method of the RandomForest class.

    Args:
        random_forest (RandomForest): Instance of the RandomForest class.

    Asserts:
        The boostrap sample has the same shape as the original dataset.
    """
    X_sample, y_sample = random_forest.bootstrap_sample(X_train, y_train)
    assert X_sample.shape == X_train.shape
    assert y_sample.shape == y_train.shape


def test_fit(random_forest: RandomForest) -> None:
    """
    Test the fit method of the RandomForest Class.

    Args:
        random_forest (RandomForest): Instance of the RandomForest class.

    Asserts:
        The number of trees in the forest should equal the number specified in the class.
        Each tree in the forest should be an instance of DecisionTree.
    """
    random_forest.fit(X_train, y_train)
    assert len(random_forest.trees) == random_forest.n_trees
    assert all(isinstance(tree, DecisionTree) for tree in random_forest.trees)


def test_predict(random_forest: RandomForest) -> None:
    """
    Test the predict method of the RandomForest Class.

    Args:
        random_forest (RandomForest): Instance of the RandomForest class.

    Asserts:
        The shape of predictions should match the number of test samples.
        The predictions should match the expected prediction values.
    """
    random_forest.fit(X_train, y_train)
    predictions = random_forest.predict(X_test)
    assert predictions.shape == (X_test.shape[0],)
    expected_predictions = np.array([0, 1])
    np.testing.assert_array_equal(predictions, expected_predictions)
