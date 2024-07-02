## Recall in Machine Learning

Recall (also known as sensitivity or true positive rate) is a metric used to evaluate the performance of a classification model, measuring the proportion of actual positives that are correctly identified.

### Formula

The formula for recall is:

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

where:
- $TP$ = True Positives
- $FN$ = False Negatives

### Worked Example

Consider the following confusion matrix for a binary classification problem:

|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | TP = 50            | FN = 10            |
| **Actual Negative** | FP = 5             | TN = 35            |

Using the values from the confusion matrix:
- True Positives (TP) = 50
- False Negatives (FN) = 10

The recall is calculated as:

$$
\text{Recall} = \frac{TP}{TP + FN} = \frac{50}{50 + 10} = \frac{50}{60} \approx 0.833
$$

This means that approximately 83.3% of the actual positive instances are correctly identified by the model.

## When is Recall Used as a Metric?

Recall is particularly useful as a metric in scenarios where it is important to identify as many positive instances as possible, even at the expense of some false positives. Here are some contexts in which recall is especially valuable:

### Scenarios Where Recall is Important

1. **Medical Diagnostics**:
   - In medical testing, it is crucial to identify as many true positive cases (e.g., detecting a disease) as possible.
   - Missing a positive case (false negative) could be more detrimental than a false alarm (false positive).
   - **Example**: Cancer screening tests prioritize recall to ensure that most cases of cancer are detected, even if it means having some false positives.

2. **Fraud Detection**:
   - In financial transactions or credit card usage, detecting fraudulent activities is critical.
   - Missing a fraudulent transaction (false negative) could result in significant financial loss, so recall is prioritized.
   - **Example**: Fraud detection systems aim to catch as many fraudulent transactions as possible, accepting some false positives as a trade-off.

3. **Spam Detection**:
   - Email or message spam filters aim to identify and block as much spam as possible.
   - Allowing a spam email to pass through (false negative) could lead to security risks or annoy the user, so recall is prioritized.
   - **Example**: Spam filters are designed to catch a high percentage of spam emails, even if it means sometimes marking legitimate emails as spam (false positives).

4. **Search Engines**:
   - Search engines aim to retrieve all relevant documents related to a user's query.
   - Missing relevant documents (false negatives) would result in an incomplete search result, so recall is prioritized.
   - **Example**: When searching for scholarly articles, a search engine prioritizes recall to ensure that most relevant articles are retrieved, even if it means including some less relevant ones.

5. **Information Retrieval Systems**:
   - In large databases or document retrieval systems, it is essential to retrieve all relevant documents related to a search query.
   - Missing relevant documents (false negatives) would mean that the user might miss out on important information.
   - **Example**: Legal document retrieval systems need to prioritize recall to ensure that all relevant legal documents are retrieved for a case.


