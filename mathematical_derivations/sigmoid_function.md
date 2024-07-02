## Sigmoid Function: A Summary

### Mathematical Definition
The sigmoid function, denoted as $\sigma(x)$, is defined as:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

### Mathematical Range
The output of the sigmoid function is always in the range (0, 1). This is because:
- $e^{-x}$ is always positive for any real number $x$, ensuring the denominator $1 + e^{-x}$ is always greater than 1.
- As $x \to \infty$, $e^{-x} \to 0$, making $\sigma(x) \to 1$.
- As $x \to -\infty$, $e^{-x} \to \infty$, making $\sigma(x) \to 0$.

Thus, $\sigma(x)$ maps any real number to a value strictly between 0 and 1.

### Application
The sigmoid function is widely used in various fields, including:
- **Neural Networks**: As an activation function that introduces non-linearity into the model, allowing it to learn complex patterns.
- **Logistic Regression**: To model the probability of a binary outcome, mapping predicted values to probabilities.
- **Probability Theory**: To describe the cumulative distribution function (CDF) of the logistic distribution.
- **Biology**: To model population growth and other processes that exhibit an initial exponential growth and a subsequent plateau.

The sigmoid function's smooth, continuous nature and bounded output make it particularly useful for scenarios where outputs need to be interpreted as probabilities.
