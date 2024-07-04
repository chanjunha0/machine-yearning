from typing import List


def calculate_standard_deviation(data: List[int]) -> float:
    """
    Calculates the standard deviation of a given dataset.

    Args:
        data (List[int]): A list of integers

    Returns:
        float: Standard deviation of the data.
    """
    # Calculate the mean
    mean = sum(data) / len(data)

    # Calculate the squared differences from the mean
    squared_diff = [(x - mean) ** 2 for x in data]

    # Calculate the variance
    variance = sum(squared_diff) / len(data)

    # Calculate the standard deviation
    std_dev = variance**0.5

    return std_dev
