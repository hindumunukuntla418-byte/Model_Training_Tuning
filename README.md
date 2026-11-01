# Supervised Classification / Regression Modeling & Tuning

## Project Overview

This project develops and evaluates multiple supervised machine learning classification models for predicting loan default.

The objective is to compare different machine learning algorithms, optimize their hyperparameters, evaluate their performance, and select the best-performing model.

## Dataset

The dataset contains information about loan applicants, including:

* Age
* Income
* Loan amount
* Credit score
* Employment type
* Education
* Housing type
* Loan purpose
* Marital status
* Years employed
* Default status

The target variable is:

`default`

Where:

* `0` = No default
* `1` = Default

## Machine Learning Models

Four classification algorithms were trained and compared:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. XGBoost

## Data Preprocessing

A leak-free preprocessing pipeline was created using Scikit-learn.

The preprocessing includes:

* Missing value imputation
* Median imputation for numerical features
* Most-frequent imputation for categorical features
* StandardScaler for numerical features
* OneHotEncoder for categorical features
* ColumnTransformer
* Scikit-learn Pipeline

The train-test split was performed before model training.

## Train-Test Split

The dataset was divided into:

* 75% training data
* 25% testing data

Stratification was used to preserve the class distribution.

## Hyperparameter Optimization

GridSearchCV was used to optimize the models.

A 5-fold StratifiedKFold cross-validation strategy was used.

The optimization metric was:

**F1-Score**

Hyperparameters were tuned separately for each model.

## Evaluation Metrics

The models were evaluated using:

* Precision
* Recall
* F1-Score
* ROC-AUC
* Classification Report
* Confusion Matrix
* ROC-AUC Curves

## Champion Model

The final champion model was selected automatically based on the highest F1-Score among the tuned models.

The selected champion model was saved using Joblib as:

`champion_model.joblib`

## Project Structure

```text
Model_Training_Tuning/
│
├── data/
│   └── loan_default_dataset.csv
│
├── notebooks/
│   └── model_training.ipynb
│
├── models/
│   └── champion_model.joblib
│
├── results/
│   ├── confusion_matrix.png
│   ├── model_comparison.csv
│   └── roc_auc_curves.png
│
└── README.md
```

## Results

The project produces a model comparison table containing:

* Precision
* Recall
* F1-Score
* ROC-AUC

The project also produces:

* Confusion matrix visualization
* ROC-AUC comparison graph
* Serialized champion machine learning model

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook

## How to Run

1. Open the project in VS Code.
2. Select the Python environment used for the project.
3. Open:

`notebooks/model_training.ipynb`

4. Run the notebook cells from top to bottom.
5. The trained model will be saved in the `models` folder.
6. Evaluation results and visualizations will be saved in the `results` folder.

## Expected Outcome

The completed project demonstrates a reproducible supervised machine learning workflow covering:

* Data preparation
* Feature preprocessing
* Multiple model training
* Hyperparameter tuning
* Stratified cross-validation
* Model evaluation
* Champion model selection
* Model serialization
* Performance visualization
