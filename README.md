# Telco Customer Churn Analysis

An end-to-end data analysis project examining why telecom customers churn and what the business can do about it. The project covers SQL analysis, Python-based machine learning, and a Power BI dashboard — built to reflect the kind of work a data analyst would actually do on this problem.

Live app: https://churnmodel-m.streamlit.app/

---

## Business Problem

Telecom companies lose significant revenue every year to churn. The cost of acquiring a new customer is typically 5 to 7 times higher than retaining an existing one, which means even a small reduction in churn rate has a meaningful impact on the bottom line.

This project answers three questions:

- Which customers are most likely to churn and why?
- How much revenue is at risk?
- What does a data-driven retention strategy look like?

---

## Dataset

IBM Telco Customer Churn dataset from Kaggle. 7,043 customers, 21 columns covering demographics, account details, subscribed services, and contract information. Target variable is whether the customer churned or not.

One data quality issue: TotalCharges was stored as a string with blank values for customers with zero tenure. Identified and corrected during cleaning.

---

## Key Findings

**Churn rate is 26.5%** — roughly 1 in 4 customers is leaving. At an average monthly charge of $64.76, that represents around $139,000 in monthly recurring revenue at risk.

The three strongest churn drivers found in the analysis:

- Contract type: month-to-month customers churn at 43%, compared to 11% for one-year and 3% for two-year contracts
- Internet service: fiber optic customers churn at 42%, more than double the rate of DSL customers (19%)
- Payment method: electronic check users churn at 45%, versus 16-18% for auto-pay customers

Customers with zero add-on services churn at 31%. Each additional add-on service reduces churn by roughly 3-5 percentage points — customers who are more embedded in the product are far less likely to leave.

New customers are the highest risk: the 0-12 month group churns at 48%, dropping sharply to around 15% for customers past 24 months.

---

## What Was Built

**SQL analysis (SQL_churn_analysis.sql)**

A complete SQL file covering table schema, data cleaning, exploratory queries, feature engineering as reusable views, and business reporting queries. Also includes a dedicated window functions section covering RANK, DENSE_RANK, PERCENT_RANK, NTILE, ROW_NUMBER, running totals, cumulative aggregates, and segment-vs-overall comparisons — the patterns that come up most in analyst interviews and day-to-day reporting work.

---

**Python ML pipeline**

A modular codebase with separate components for data ingestion, feature engineering, model training, and prediction. Eight classification models were trained and compared. Logistic Regression performed best with a ROC-AUC of 0.847. The model was deployed as an interactive Streamlit app where you can input customer details and get a churn probability with a risk level and explanation of which risk factors are present.

One finding worth noting: the default prediction threshold of 0.5 only caught 54% of actual churners. After diagnosing this post-deployment, the threshold was tuned to 0.35, which improved recall to 71% — a meaningful difference in a retention campaign context.

---

## SQL Highlights

The SQL file is structured in seven sections. A few queries worth calling out:

Churn rate by segment vs overall in a single pass using window aggregates — no subquery or self-join needed:

---

## Machine Learning Summary

| Model               | ROC-AUC | Recall (at 0.35) |
|---------------------|---------|-----------------|
| Logistic Regression | 0.847   | 71.1%           |
| AdaBoost            | 0.845   | 68.7%           |
| Gradient Boosting   | 0.838   | 66.2%           |

ROC-AUC was chosen as the primary metric because the dataset has a 73/27 class split. A model that always predicts "no churn" would get 73.5% accuracy — ROC-AUC correctly identifies this as no better than random.

---

## Business Recommendation

Based on the analysis, a targeted retention campaign should prioritise:

1. Month-to-month customers on fiber optic internet paying by electronic check — this combination has the highest observed churn rate
2. Customers in their first 12 months — the drop-off in churn after month 12 suggests early engagement is critical
3. Customers with zero add-on services — each service added is associated with meaningfully lower churn

The retention ROI query in the SQL file estimates the net value of a retention campaign based on configurable cost and revenue assumptions.

## Tools Used

Python, scikit-learn, pandas, numpy, seaborn, matplotlib, Streamlit, joblib, SQL, Power BI
