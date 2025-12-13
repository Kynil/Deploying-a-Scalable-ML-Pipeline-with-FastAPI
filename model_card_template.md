# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is a supervised binary classification model implemented using a RandomForestClassifier. It predicts whether an individual's income is greater than 50K or less than or equal to 50K based on demographic and employment related census features. The model uses one-hot encoding for categorical features and outputs a binary prediction for the salary label.

## Intended Use

The model is intended for educational purposes as part of a learning project. It demonstrates data preprocessing, model training, evaluation, slice-based performance analysis and deployment via REST API.

## Training Data

The training data comes from the census dataset providied in data/census.csv. The target variable is salary, indicating whether an individual earns more than 50k annually. The dataset contains a mix of numeric and categorial features including workclass, education, marital status, occupation, relationship, sex, race, and native country. The data is split into training and test sets using an 80/20 split with stratification on the target label.

## Evaluation Data

The evaluation data consists of the held-out test split that was not used during training. The same preprocessing pipeline and fitted encoders from the training phase are reused during evaluation to ensure consistency.

## Metrics

The model is evaluated using precision, recall, and F1 score (F-beta score with beta =1).
Overall performance:
Precision: 0.7338 | Recall: 0.6365 | F1: 0.6817

## Ethical Considerations

The dataset includes demographic attributes such as race and sex, and as such may present biases in the historical data. This dataset and model is for learning purposes and as such should not be used in real-world settings, but if it is, the model could contribute to unfair or discriminatory outcomes due to possible bias.

## Caveats and Recommendations

Model performance varies across data slices, particularly for categories with very small sample sizes, where metrics could be unstable or unreliable. The model's predictions are dependent on quality and representation of the training data.