# Credit Card Balance Prediction

A regression project that predicts the average credit card balance of a customer
from financial and personal details.

## Problem
Banks need to understand which customers carry high balances.
I built a model to predict `Balance` (a dollar amount) for 400 customers.

## Data
400 customers, 10 features (income, credit limit, rating, cards, age, education, and others).
No missing values. 22% of customers have a balance of exactly 0, mostly those with a low credit limit.

## Steps
1. Exploratory analysis: correlations, group comparisons, the zero-balance pattern
2. Multicollinearity: Limit and Rating were almost identical (correlation 0.997, VIF above 230), so I dropped Rating
3. Preparation: train/test split (stratified on zero balances), scaling and encoding in a pipeline
4. Modelling: baseline, linear regression, ridge, random forest and gradient boosting (5-fold cross-validation)
5. Tuning: two rounds of grid search on gradient boosting
6. Evaluation on 80 unseen customers

## Results

Cross-validation on the training set:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Baseline (average) | 467.5 | 398.2 | -0.017 |
| Linear Regression | 103.9 | 84.5 | 0.949 |
| Ridge | 104.4 | 85.9 | 0.949 |
| Random Forest | 119.3 | 72.9 | 0.933 |
| Gradient Boosting (default) | 100.0 | 65.1 | 0.953 |
| **Gradient Boosting (tuned)** | **about 74** | n/a | n/a |

Final test on 80 unseen customers (tuned gradient boosting):

| RMSE | MAE | R² |
|---|---|---|
| [about 57] | [about 41] | [about 0.98] |

The model is off by about $[41] on average.

## Key findings
- Credit limit is by far the strongest predictor, then income, then whether the customer is a student.
- Age, education, marital status, home ownership and region added almost nothing.
- Customers with low credit limits mostly have a zero balance. A straight-line model cannot capture this,
  so tree-based models did better.
- Shallow trees (depth 2) worked best, which matches the simple threshold pattern in the data.
- A few predictions are slightly below zero, so I set them to 0 in practice.

## Limitations
- Small dataset (400 customers); the test set has only 80, so results are an estimate.
- The tuned model uses fairly extreme settings, and I stopped tuning after two rounds to avoid overfitting.
- Gradient boosting is harder to explain than linear regression.
- This is a learning dataset and must not be used for real credit decisions.

## How to run
1. Clone this repository
2. Install: `pip install pandas scikit-learn matplotlib seaborn joblib`
3. Put `Credit_xlsx.csv` next to the notebook (the file uses semicolons, so load it with `sep=";"`)
4. Open `credit_balance_prediction.ipynb` and run all cells

## Tools
Python, pandas, scikit-learn, matplotlib, seaborn

## Author
Kefentse Malatji
