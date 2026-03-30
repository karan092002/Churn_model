import streamlit as st
import pandas as pd
import sys
import os

# Make src importable when running from the project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.predict_pipeline import PredictPipeline, CustomerData
from src.exception import CustomException

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("Customer Churn Predictor")
st.write("Fill in the customer details below to get a churn probability.")

# Input form
st.header("Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 800.0)
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

with col2:
    contract = st.selectbox("Contract", [
        "Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", [
                          "DSL", "Fiber optic", "No"])
    payment = st.selectbox("Payment Method", [
                          "Electronic check", "Mailed check",
                          "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

with col3:
    gender = st.selectbox("Gender", ["Male", "Female"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

st.subheader("Add-on Services")
addon_col1, addon_col2, addon_col3 = st.columns(3)

with addon_col1:
    multiple_lines = st.selectbox("Multiple Lines",     ["No", "Yes", "No phone service"])
    online_security = st.selectbox("Online Security",    ["No", "Yes", "No internet service"])

with addon_col2:
    online_backup = st.selectbox("Online Backup",      ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection",  ["No", "Yes", "No internet service"])

with addon_col3:
    tech_support = st.selectbox("Tech Support",       ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV",       ["No", "Yes", "No internet service"])

streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict Churn", type="primary"):
    try:
        customer = CustomerData(
            gender = gender,
            senior_citizen = 1 if senior_citizen == "Yes" else 0,
            partner = partner,
            dependents = dependents,
            tenure = tenure,
            phone_service = phone_service,
            multiple_lines = multiple_lines,
            internet_service = internet,
            online_security = online_security,
            online_backup = online_backup,
            device_protection = device_protection,
            tech_support = tech_support,
            streaming_tv = streaming_tv,
            streaming_movies = streaming_movies,
            contract = contract,
            paperless_billing = paperless,
            payment_method = payment,
            monthly_charges = monthly_charges,
            total_charges = total_charges,
        )

        pipeline = PredictPipeline()
        prob = pipeline.predict(customer.to_dataframe())[0]
        label = "Will Churn" if prob >= 0.5 else "Will Stay"

        st.divider()
        st.header("Result")

        col_a, col_b = st.columns(2)
        col_a.metric("Prediction", label)
        col_b.metric("Churn Probability", f"{prob:.1%}")

        st.progress(float(prob))

        if prob >= 0.5:
            st.error("High churn risk — consider a retention offer.")
        else:
            st.success("Low churn risk — customer likely to stay.")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
