# Algorithms

Welcome to the Algorithms repository! This repository logs my explorations, curiosity, and learnings in the domain of algorithms. Here, you will find various algorithms and key functions that I have studied, along with detailed explanations and implementations.

## Summary Table of Explored Algorithms and Key Functions

| Name                     | Type                  | Logic                 | Description                                                                 | Method                | Normalization Needed |
|--------------------------|-----------------------|-----------------------|-----------------------------------------------------------------------------|-----------------------|-----------------------|
| Decision Tree            | Machine Learning      | Recursive Partitioning| A tree-like model of decisions based on feature values, used for classification and regression. | Gini Impurity         | No                    |
| RandomForest             | Machine Learning      | Ensemble Learning     | A collection of decision trees, typically trained with the bagging method, used for classification and regression. | Majority Voting / Averaging | No                    |
| Logistic Regression      | Machine Learning      | Optimization          | An algorithm for binary classification using a logistic function.           | Gradient Descent      | Yes                   |
| Sigmoid Function         | Mathematical Function | Transformation        | A function that maps any real-valued number to the range (0, 1). Used in logistic regression and neural networks. | Mathematical Formula  | No                    |
| Sum of Squared Residuals | Statistical Measure   | Error Measurement     | A measure of the discrepancy between the observed data and the values predicted by a model. | Residual Calculation  | No                    |
| Standardize Column       | Data Preprocessing    | Normalization         | Standardizes the column of a given dataset to have a mean of 0 and a standard deviation of 1. | Feature Scaling       | Yes                   |
| Standard Deviation       | Statistical Measure   | Dispersion Measurement| A measure of the amount of variation or dispersion in a set of values.       | Square Root of Variance | No                    |



## Algorithm Concept (High Level)

## Logistic Regression
1. Prepare, train test split
2. Initialise Weights and Bias
3. Compute Linear Combination
4. Sigmoid Function for Probabilites
5. Compute binary cross-entropy
6. Update weights with gradient descent


## Decision Tree
1. Prepare, train test split
2. Start at root, recursively split based on best feature and threshold
3. Use splitting criteria like Gini impurity or entrophy
4. Stopping criteria (max depth or min split)


## Random Forest
1. Prepare, train test split
2. Create multiple bootstrap samples 
3. Build decision tree for each sample
4. Classification - use majority voting
5. Regression - average predictions across trees

## Mathematical Formulas 

## Gini Impurity
For a dataset with $n$ classes, the Gini impurity is defined as:

$$
\text{Gini} = 1 - \sum_{i=1}^{n} p_i^2
$$

where $p_i$ is the proportion/probability of elements belonging to class $i$ in the dataset.