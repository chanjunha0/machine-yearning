import numpy as np


def sigmoid(z):
    """
    Computes the sigmoid of z

    Args:
        z (float or np.ndarray): The input value or array of values.

    Returns:
        float or np.ndarray: The sigmoid if the input values(s), which is always between 0 and 1

    Explanation:
        Given an input z, the sigmoid function returns a value between 0 and 1.
        This is useful in binary classification, where the output can be interpreted as a probability.

        Mathematically:
            sigmoid(z) = 1 / (1 + exp(-z))

        Where:
            - 'exp' is the exponential function.
            - 'z' is the input value which can be any real number.
            - The output is a real number between 0 and 1.

        Example:
            If z = 0, sigmoid(0) = 1 / (1 + exp(0)) = 1 / 2 = 0.5
            If z is very large, sigmoid(z) approaches 1.
            If z is very small (negative), sigmoid(z) approaches 0.

        Reference Links:
            - https://www.sciencedirect.com/topics/computer-science/sigmoid-function#:~:text=The%20sigmoid%20function%20transforms%20the,the%20weight%20is%20more%20stable.

    Additional Notes:
        - One of the most commonly used activation functions
    """
    return 1 / (1 + np.exp(-z))
