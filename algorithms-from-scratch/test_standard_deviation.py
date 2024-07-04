import pytest
import numpy as np

from standard_deviation import calculate_standard_deviation


def test_standard_deviation():
    """
    Test the standard deviation method.
    """
    data = [10, 12, 23, 23, 16, 23, 21, 16]
    std_dev_custom = calculate_standard_deviation(data)
    std_dev_np = np.std(data)
    assert np.isclose(std_dev_custom, std_dev_np)
