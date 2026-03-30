# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn. The goal was to build a complete pipeline from raw data to a deployed app, not just train a model in a notebook.

Live demo: https://your-app.streamlit.app

---

## What this project is about

Telecom companies lose a significant portion of their revenue every year to churn. The problem is that by the time a customer cancels, it's too late. This model tries to identify at-risk customers early enough that the business can intervene with a retention offer.

The dataset is the IBM Telco Customer Churn dataset from Kaggle — around 7,000 customers with information about their plan, contract type, payment method, and usage. The target variable is whether they churned or not.

One thing worth noting upfront: only about 26.5% of customers in the dataset churned. That imbalance matters a lot for how you evaluate the model, which is why I used ROC-AUC as the primary metric instead of accuracy.

---

## Results

I trained and compared 8 models. Here are the results on the held-out test set:

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------------------|----------|-----------|--------|--------|---------|
| Logistic Regression | 0.8091   | 0.6733    | 0.5455 | 0.6027 | 0.8466  |
| AdaBoost            | 0.7999   | 0.6513    | 0.5294 | 0.5841 | 0.8447  |
| Random Forest       | 0.7842   | 0.6167    | 0.4947 | 0.5490 | 0.8190  |
| Decision Tree       | 0.7913   | 0.6266    | 0.5294 | 0.5739 | 0.8329  |
| KNN                 | 0.7544   | 0.5417    | 0.4866 | 0.5127 | 0.7928  |
| SVM                 | 0.7921   | 0.6421    | 0.4893 | 0.5554 | 0.7876  |

Logistic Regression came out on top with an AUC of 0.8466 and the most consistent cross-validation scores across 5 folds (mean 0.8488, std 0.012). Interestingly the more complex models like Random Forest didn't do better — probably because the feature engineering created clean enough signals that a linear model could pick up directly.

---

## Pipeline overview

The notebook walks through each of these steps in order:

1. EDA — distributions, missing values, churn rate by contract type, payment method, tenure, etc.
2. Data cleaning — TotalCharges was stored as a string with blank values for new customers, needed coercion to numeric and median imputation
3. Feature engineering — created 8 new features from existing columns (details below)
4. Train/test split — 80/20 stratified split to preserve the churn ratio in both sets
5. Preprocessing — median imputation and standard scaling inside a sklearn Pipeline so nothing leaks into the test set
6. Model comparison — 8 models evaluated with 5-fold cross-validation
7. Evaluation — confusion matrices, ROC curves, feature importance, classification report

---

## Features I created

Most of these came from looking at the EDA and noticing patterns:

- AvgMonthlySpend: TotalCharges divided by tenure. Raw total charges are misleading because long-tenured customers naturally have higher totals, this normalises it.
- NumAddOns: count of how many add-on services the customer has. Customers with more add-ons are more invested and less likely to leave.
- HasStreaming: whether the customer has either streaming service. Combines two weak columns into one signal.
- HasOnlineServices: whether the customer has any of the four protection or support services.
- IsMonthToMonth: binary flag for month-to-month contract. This was the clearest pattern in EDA by far.
- HasFiberOptic: fiber optic customers churned disproportionately, likely due to pricing.
- IsElectronicCheck: electronic check users showed notably higher churn than customers on auto-pay.
- TenureGroup: tenure bucketed into four bands (0-12m, 13-24m, 25-48m, 49-72m) to capture the non-linear relationship between tenure and churn.

---

## A note on data leakage

All preprocessing is done inside a sklearn Pipeline object, meaning the scaler and imputer are fit only on training data and then applied to the test set. If you fit them on the full dataset before splitting, the test set's distribution influences the transformation parameters — that's leakage and leads to overly optimistic evaluation scores.