-- CUSTOMER CHURN — SQL ANALYSIS

-- This file covers:
--   1. Table schema
--   2. Data cleaning
--   3. Exploratory queries (mirrors EDA from the notebook)
--   4. Feature engineering as SQL views
--   5. Model-ready export query
--   6. Business reporting queries
--   7. Window functions


-- 1. TABLE SCHEMA

create database customer_churn;
use customer_churn;


SELECT COUNT(*)
from churn_table;

select * from churn_table limit 5;


-- 2. DATA CLEANING

-- TotalCharges is blank for customers with tenure = 0
-- Update those rows with a value of 0

update churn_table
set TotalCharges = 0
where TotalCharges is null;

-- Verify no null values remain
SELECT COUNT(*) FROM churn_table where TotalCharges is null;


-- 3. EXPLORATORY QUERIES

-- 3.1 Overall churn rate
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    SUM(CASE WHEN Churn = 'No' THEN 1 ELSE 0 END) AS retained,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    2) AS churn_rate_pct
FROM churn_table;

SELECT * FROM churn_table;


-- 3.2 Churn rate by contract type
SELECT
    Contract,
    COUNT(*) AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    2) AS churn_rate_pct
FROM churn_table
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- 3.3 Churn rate by internet service type
SELECT
    InternetService,
    COUNT(*) AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    2) AS churn_rate_pct
FROM churn_table
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;


-- 3.4 Churn rate by payment method
SELECT
    PaymentMethod,
    COUNT(*) AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    2) AS churn_rate_pct
FROM churn_table
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;


-- 3.5 Churn rate by senior citizen status
SELECT
    CASE WHEN SeniorCitizen = 1 THEN 'Senior' ELSE 'Non-Senior' END AS segment,
    COUNT(*) AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    2) AS churn_rate_pct
FROM churn_table
GROUP BY SeniorCitizen;


-- 3.6 Average tenure and charges by churn status
SELECT
    Churn,
    ROUND(AVG(tenure), 2) AS avg_tenure_months,
    ROUND(AVG(MonthlyCharges), 2)  AS avg_monthly_charges,
    ROUND(AVG(TotalCharges), 2) AS avg_total_charges,
    ROUND(MIN(MonthlyCharges), 2) AS min_monthly_charges,
    ROUND(MAX(MonthlyCharges), 2) AS max_monthly_charges
FROM churn_table
GROUP BY Churn
ORDER BY Churn DESC;


-- 3.7 Churn by tenure bucket
SELECT * FROM churn_table;

SELECT
    CASE
        WHEN tenure BETWEEN 0  AND 12 THEN '0-12 months'
        WHEN tenure BETWEEN 13 AND 24 THEN '13-24 months'
        WHEN tenure BETWEEN 25 AND 48 THEN '25-48 months'
        WHEN tenure BETWEEN 49 AND 72 THEN '49-72 months'
    END AS tenure_group,
    COUNT(*) AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    2) AS churn_rate_pct
FROM churn_table
GROUP BY tenure_group
ORDER BY MIN(tenure);


-- 3.8 Number of add-ons vs churn rate
SELECT
    num_addons,
    COUNT(*) AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    2) AS churn_rate_pct
FROM (
    SELECT
        Churn,
        (
            CASE WHEN OnlineSecurity = 'Yes' THEN 1 ELSE 0 END +
            CASE WHEN OnlineBackup = 'Yes' THEN 1 ELSE 0 END +
            CASE WHEN DeviceProtection = 'Yes' THEN 1 ELSE 0 END +
            CASE WHEN TechSupport = 'Yes' THEN 1 ELSE 0 END +
            CASE WHEN StreamingTV = 'Yes' THEN 1 ELSE 0 END +
            CASE WHEN StreamingMovies = 'Yes' THEN 1 ELSE 0 END
        ) AS num_addons
    FROM churn_table
) t
GROUP BY num_addons
ORDER BY num_addons;


-- 3.9 Monthly revenue at risk from churners
SELECT
    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_revenue,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) AS revenue_at_risk,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END) * 100.0/ SUM(MonthlyCharges),2)
        AS pct_revenue_at_risk
FROM churn_table;


-- 3.10 Cross tabulation — Contract vs Internet Service churn rates
SELECT
    Contract,
    InternetService,
    COUNT(*) AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),2) 
        AS churn_rate_pct
FROM churn_table
GROUP BY Contract, InternetService
ORDER BY churn_rate_pct DESC;


-- 4. FEATURE ENGINEERING AS SQL VIEWS

CREATE OR REPLACE VIEW vw_telco_engineered AS
SELECT
    customerID,
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    MonthlyCharges,
    TotalCharges,
    Churn,

    ROUND(TotalCharges / NULLIF(tenure + 1, 0), 2) AS AvgMonthlySpend,
    (
        CASE WHEN OnlineSecurity = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN OnlineBackup = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN DeviceProtection = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN TechSupport = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN StreamingTV = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN StreamingMovies = 'Yes' THEN 1 ELSE 0 END
    ) AS NumAddOns,
    CASE WHEN StreamingTV = 'Yes'
          OR StreamingMovies = 'Yes'
         THEN 1 ELSE 0 END AS HasStreaming,

    CASE WHEN OnlineSecurity = 'Yes'
          OR OnlineBackup = 'Yes'
          OR DeviceProtection = 'Yes'
          OR TechSupport = 'Yes'
         THEN 1 ELSE 0 END AS HasOnlineServices,

    CASE WHEN Contract = 'Month-to-month' THEN 1 ELSE 0 END AS IsMonthToMonth,
    CASE WHEN InternetService = 'Fiber optic' THEN 1 ELSE 0 END AS HasFiberOptic,
    CASE WHEN PaymentMethod = 'Electronic check' THEN 1 ELSE 0 END AS IsElectronicCheck,

    CASE
        WHEN tenure BETWEEN 0  AND 12 THEN '0-12m'
        WHEN tenure BETWEEN 13 AND 24 THEN '13-24m'
        WHEN tenure BETWEEN 25 AND 48 THEN '25-48m'
        WHEN tenure BETWEEN 49 AND 72 THEN '49-72m'
    END AS TenureGroup,

    CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END AS ChurnFlag

FROM churn_table;


CREATE OR REPLACE VIEW vw_churn_by_segment AS
SELECT
    Contract,
    InternetService,
    PaymentMethod,
    CASE WHEN SeniorCitizen = 1 THEN 'Senior' ELSE 'Non-Senior' END AS AgeSegment,
    CASE
        WHEN tenure BETWEEN 0 AND 12 THEN '0-12m'
        WHEN tenure BETWEEN 13 AND 24 THEN '13-24m'
        WHEN tenure BETWEEN 25 AND 48 THEN '25-48m'
        WHEN tenure BETWEEN 49 AND 72 THEN '49-72m'
    END AS TenureGroup,
    COUNT(*) AS TotalCustomers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS ChurnedCustomers,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    2) AS ChurnRatePct,
    ROUND(SUM(MonthlyCharges), 2) AS TotalMonthlyRevenue,
    ROUND(AVG(MonthlyCharges), 2) AS AvgMonthlyCharges,
    ROUND(AVG(tenure), 1) AS AvgTenure
FROM churn_table
GROUP BY
    Contract,
    InternetService,
    PaymentMethod,
    SeniorCitizen,
    TenureGroup;


-- 5. MODEL-READY EXPORT QUERY

SELECT *
FROM vw_telco_engineered
ORDER BY customerID;


-- 6. BUSINESS REPORTING QUERIES
-- 6.1 Monthly revenue at risk by contract type
SELECT
    Contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_at_risk,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN MonthlyCharges END), 2) AS avg_charges_churner
FROM churn_table
GROUP BY Contract
ORDER BY monthly_revenue_at_risk DESC;


-- 6.2 High risk customer list — top candidates for retention offers
SELECT
    customerID,
    tenure,
    Contract,
    InternetService,
    PaymentMethod,
    MonthlyCharges,
    TotalCharges,
    Churn
FROM churn_table
WHERE Contract = 'Month-to-month'
  AND InternetService = 'Fiber optic'
  AND PaymentMethod = 'Electronic check'
  AND tenure < 12
ORDER BY MonthlyCharges DESC;


-- 6.3 Customer lifetime value comparison — churners vs retained
SELECT
    Churn,
    COUNT(*) AS customers,
    ROUND(AVG(tenure), 1) AS avg_tenure_months,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
    ROUND(AVG(TotalCharges), 2) AS avg_lifetime_value,
    ROUND(SUM(TotalCharges), 2) AS total_lifetime_value
FROM churn_table
GROUP BY Churn;


-- 6.4 Retention offer ROI estimate
SELECT
    customerID,
    MonthlyCharges,
    tenure,
    Contract,
    ROUND(MonthlyCharges * 12, 2) AS projected_12mo_value,
    ROUND((MonthlyCharges * 12) - 50, 2) AS roi_if_retained,
    CASE
        WHEN (MonthlyCharges * 12) - 50 > 0
        THEN 'Worth offering'
        ELSE 'Not worth offering'
    END AS recommendation
FROM churn_table
WHERE Churn = 'Yes'
ORDER BY roi_if_retained DESC;


-- 6.5 Churn trend by tenure
SELECT
    tenure,
    COUNT(*) AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),2)
        AS churn_rate_pct
FROM churn_table
GROUP BY tenure
ORDER BY tenure;



-- 7. WINDOW FUNCTIONS
-- 7.1 RANK customers by MonthlyCharges within each contract type
-- Useful for identifying the highest-value customers at risk per segment
SELECT
    customerID,
    Contract,
    MonthlyCharges,
    Churn,
    RANK() OVER (
        PARTITION BY Contract
        ORDER BY MonthlyCharges DESC
    ) AS charges_rank_in_contract,
    DENSE_RANK() OVER (
        ORDER BY MonthlyCharges DESC
    ) AS overall_charges_rank
FROM churn_table
ORDER BY Contract, charges_rank_in_contract;


-- 7.2 RUNNING TOTAL of revenue grouped by contract type ordered by tenure
-- Shows how cumulative revenue builds up as customers stay longer
SELECT
    customerID,
    Contract,
    tenure,
    MonthlyCharges,
    SUM(MonthlyCharges) OVER (
        PARTITION BY Contract
        ORDER BY tenure
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_revenue_in_contract,
    SUM(MonthlyCharges) OVER (
        ORDER BY tenure
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS overall_running_revenue
FROM churn_table
ORDER BY tenure;


-- 7.3 COMPARE each customer's monthly charges to their contract segment average
-- Flags customers paying above average for their contract type
SELECT
    customerID,
    Contract,
    MonthlyCharges,
    Churn,
    ROUND(AVG(MonthlyCharges) OVER (
        PARTITION BY Contract
    ), 2) AS avg_charges_in_contract,
    ROUND(MonthlyCharges - AVG(MonthlyCharges) OVER (
        PARTITION BY Contract
    ), 2) AS diff_from_contract_avg,
    ROUND(AVG(MonthlyCharges) OVER (), 2)
    AS overall_avg_charges,
    ROUND(MonthlyCharges - AVG(MonthlyCharges) OVER (), 2)
    AS diff_from_overall_avg
FROM churn_table
ORDER BY diff_from_contract_avg DESC;


-- 7.4 PERCENTILE — where does each customer fall in the charges distribution
-- within their internet service type
SELECT
    customerID,
    InternetService,
    MonthlyCharges,
    Churn,
    ROUND(
        PERCENT_RANK() OVER (
            PARTITION BY InternetService
            ORDER BY MonthlyCharges
        ) * 100,1)
        AS charges_percentile,
    NTILE(4) OVER (
        PARTITION BY InternetService
        ORDER BY MonthlyCharges
    ) AS charges_quartile  -- 1=lowest, 4=highest
FROM churn_table
ORDER BY InternetService, charges_percentile;



-- 7.5 ROW_NUMBER to deduplicate — useful if the dataset ever has duplicates
-- or when selecting one row per customer from a larger activity table
SELECT *
FROM (
    SELECT
        customerID,
        Contract,
        MonthlyCharges,
        Churn,
        ROW_NUMBER() OVER (
            PARTITION BY customerID
            ORDER BY MonthlyCharges DESC
        ) AS rn
    FROM churn_table
) ranked
WHERE rn = 1;


-- 7.7 CHURN RATE per contract vs overall — side by side using window function
-- No subquery or JOIN needed — window aggregates the overall in the same pass
SELECT
    Contract,
    COUNT(*) AS segment_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS segment_churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0/ COUNT(*),2)
        AS segment_churn_rate_pct,
    ROUND(
        SUM(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)) OVER () * 100.0/ SUM(COUNT(*)) OVER (),2)
        AS overall_churn_rate_pct,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)
        - SUM(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)) OVER () * 100.0
          / SUM(COUNT(*)) OVER (),2)
          AS diff_from_overall_pct
FROM churn_table
GROUP BY Contract
ORDER BY segment_churn_rate_pct DESC;


-- 7.8 CUMULATIVE CHURN COUNT ordered by MonthlyCharges
-- Shows how many churners are captured as you move up the revenue ladder
SELECT
    customerID,
    MonthlyCharges,
    Churn,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) OVER (
        ORDER BY MonthlyCharges
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_churners,
    COUNT(*) OVER (
        ORDER BY MonthlyCharges
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_customers,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) OVER (
            ORDER BY MonthlyCharges
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) * 100.0 /
        COUNT(*) OVER (
            ORDER BY MonthlyCharges
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),2) AS cumulative_churn_rate_pct
FROM churn_table
ORDER BY MonthlyCharges;




















