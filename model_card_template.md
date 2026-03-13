# Model Card

## Model Details
This model predicts whether an individual's income exceeds $50K per year using demographic and     employment-related features from the U.S. Census dataset. The model is implemented using Python and the scikit-learn machine learning library.

A RandomForestClassifier algorithm is used for training due to its strong performance on tabular datasets and its ability to capture nonlinear relationships between variables. The model is trained on processed data where categorical features are transformed using one-hot encoding and the target label is binarized.

The model is part of a machine learning pipeline deployed through a FastAPI application to support scalable inference.

The model achieved an F1 score of 0.6863 on the held-out test dataset.

## Intended Use
The model is intended for educational and experimental purposes, specifically to demonstrate the deployment of a machine learning pipeline using FastAPI. It predicts whether an individual's income exceeds $50K annually based on demographic and employment-related attributes.

This model should not be used for real-world decision making related to employment, hiring, financial approval, or policy decisions without significant additional validation, fairness analysis, and monitoring.

## Training Data
The training data comes from the UCI Census Income dataset, which contains demographic and employment information derived from the 1994 U.S. Census database.

Features include attributes such as:
- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Hours per week
- Native country

Categorical features are encoded using one-hot encoding during preprocessing. The dataset is split into training and testing sets using an 80/20 split.

## Evaluation Data
The evaluation data consists of the 20% holdout test dataset generated during the train-test split. This dataset is processed using the same preprocessing pipeline as the training data, including the previously fitted encoder and label binarizer.

Slice-based evaluation is also performed on categorical features to analyze model performance across different demographic groups.

## Metrics
The model is evaluated using the following classification metrics:
- Precision
- Recall
- F1 Score (F-beta score with beta=1)

Performance on the held-out test dataset:
- Precision: 0.7419
- Recall: 0.6384
- F1 Score: 0.6863

These metrics measure the model’s ability to correctly predict whether an individual earns more than $50K annually. Precision indicates the proportion of positive predictions that are correct, recall measures how many actual positive cases were identified, and the F1 score provides a balance between precision and recall.

In addition to overall evaluation metrics, slice-based analysis is performed on categorical features. Metrics for each slice are recorded in `slice_output.txt` to help identify performance differences across demographic groups.

## Ethical Considerations
This model uses demographic and socioeconomic attributes that may reflect historical biases present in the underlying census data. As a result, the model may produce predictions that unintentionally reinforce or reflect societal inequalities.

Certain features such as race, sex, and marital status are sensitive attributes that could lead to biased predictions if used in real-world decision systems. Care must be taken when interpreting results, and additional fairness analysis would be necessary before deploying this model in a production environment.

## Caveats and Recommendations
This model was trained for demonstration and educational purposes and has several limitations:
- The dataset is relatively old and may not reflect current socioeconomic conditions.
- The model has not been optimized with extensive hyperparameter tuning.
- Potential bias in the data may affect predictions for certain demographic groups.
- The model has not undergone fairness audits, bias mitigation, or continuous monitoring.

Future improvements could include:
- Performing fairness evaluations across demographic groups
- Applying bias mitigation techniques
- Implementing cross-validation and hyperparameter optimization
- Monitoring model performance after deployment
