import pytest
import numpy as np
from typing import Tuple
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SKLearnLogisticRegression
from sklearn.metrics import accuracy_score

from logistic_regression import LogisticRegression


@pytest.fixture
def generate_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data for binary classification

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
    """
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def test_logistic_regression(
    generate_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """
    Test the custom Logistic Regression implementation against scikit-learn's.

    Args:
        generate_data (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): Fixture providing training and test data.
    """
    X_train, X_test, y_train, y_test = generate_data

    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    custom_accuracy = accuracy_score(y_test, y_pred)

    sk_model = SKLearnLogisticRegression(random_state=42)
    sk_model.fit(X_train, y_train)
    sk_y_pred = sk_model.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sk_y_pred)

    assert (
        abs(custom_accuracy - sklearn_accuracy) < 0.1
    ), "Custom implementation accuracy differs significantly from scikit-learn"
