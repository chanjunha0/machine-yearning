## Precision in Machine Learning

Precision is a metric used to evaluate the accuracy of a classification model, specifically the proportion of positive identifications that are actually correct.

### Formula

The formula for precision is:

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

where:
- $TP$ = True Positives
- $FP$ = False Positives

### Worked Example

Consider the following confusion matrix for a binary classification problem:

|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | TP = 50            | FN = 10            |
| **Actual Negative** | FP = 5             | TN = 35            |

Using the values from the confusion matrix:
- True Positives (TP) = 50
- False Positives (FP) = 5

The precision is calculated as:

$$
\text{Precision} = \frac{TP}{TP + FP} = \frac{50}{50 + 5} = \frac{50}{55} \approx 0.909
$$

This means that approximately 90.9% of the instances predicted as positive are actually positive.
