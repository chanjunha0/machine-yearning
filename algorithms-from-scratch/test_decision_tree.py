import pytest
import numpy as np

from numpy.typing import NDArray
from .decision_tree import DecisionTree, Node

# Constants
class0: list[int] = [0, 0, 0, 0]
class1: list[int] = [1, 1, 1, 1]
X_train: NDArray[np.int_] = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y_train: NDArray[np.int_] = np.array([0, 0, 1, 1, 1])


@pytest.fixture
def decision_tree() -> DecisionTree:
    """
    Fixture for creating a DecisionTree instance with max_depth=3.

    Returns:
        DecisionTree: An instance of the DecisionTree class.
    """
    return DecisionTree(max_depth=3)


def test_gini(decision_tree: DecisionTree) -> None:
    """
    Test the gini method of the DecisionTree class.

    Args:
        decision_tree (DecisionTree): A fixture that provides an instance of the DecisionTree class.

    Asserts:
        The calculated Gini impurity should be approximately equal to the expected value.
    """
    gini_impurity = decision_tree.gini(class0, class1)
    expected_gini = 0.5
    assert pytest.approx(gini_impurity, 0.00001) == expected_gini


def test_fit(decision_tree: DecisionTree) -> None:
    """
    Test the fit method of the DecisionTree class.

    Args:
        decision_tree (DecisionTree): A fixture that provides an instance of the DecisionTree class.

    Asserts:
        The tree attribute of the DecisionTree instance should not be None after fitting.
        The tree attribute should be an instance of the Node class.

    """
    decision_tree.fit(X_train, y_train)
    assert decision_tree.tree is not None
    assert isinstance(decision_tree.tree, Node)


def test_predict(decision_tree: DecisionTree) -> None:
    """
    Test the predict method of the DecisionTree class.

    Args:
        decision_tree (DecisionTree): A fixture that provides an instance of the DecisionTree class.

    Asserts:
        The predictions made by the Decision Tree instance should match the expected predictions.
    """
    decision_tree.fit(X_train, y_train)
    X_test = np.array([[2, 3], [6, 7]])
    predictions = decision_tree.predict(X_test)
    expected_predictions = np.array([0, 1])
    np.testing.assert_array_equal(predictions, expected_predictions)
