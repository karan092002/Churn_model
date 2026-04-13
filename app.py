import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.predict_pipeline import PredictPipeline, CustomerData
from src.exception import CustomException

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("Customer Churn Predictor")
st.write("Fill in the customer details below to get a churn probability.")

# Threshold used for the final prediction label.
# 0.35 is used instead of the default 0.5 because the model has low recall
# at 0.5 — it only catches 54% of real churners. At 0.35 it catches 71%.
# The trade-off is more false positives, but for a churn use case missing
# a churner is more costly than a wasted retention offer.
THRESHOLD = 0.35

# ── Input form ────────────────────────────────────────────────────────────────
st.header("Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure (months)", 1, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                       float(65.0 * 12))
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

with col2:
    contract = st.selectbox("Contract", [
                          "Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", [
                          "Fiber optic", "DSL", "No"])
    payment = st.selectbox("Payment Method", [
                          "Electronic check", "Mailed check",
                          "Bank transfer (automatic)",
                          "Credit card (automatic)"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

with col3:
    gender = st.selectbox("Gender", ["Male", "Female"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

st.subheader("Add-on Services")

# The value "No internet service" is a valid category the model was
# trained on — use it when internet = No, otherwise use Yes/No
no_internet = internet == "No"
addon_options = ["No internet service"] if no_internet else ["No", "Yes"]

addon_col1, addon_col2, addon_col3 = st.columns(3)

with addon_col1:
    if phone_service == "No":
        multiple_lines = "No phone service"
        st.selectbox("Multiple Lines", ["No phone service"], disabled=True)
    else:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])

    online_security = st.selectbox("Online Security", addon_options)

with addon_col2:
    online_backup     = st.selectbox("Online Backup",     addon_options)
    device_protection = st.selectbox("Device Protection", addon_options)

with addon_col3:
    tech_support     = st.selectbox("Tech Support",      addon_options)
    streaming_tv     = st.selectbox("Streaming TV",      addon_options)

streaming_movies = st.selectbox("Streaming Movies", addon_options)

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
        prob     = pipeline.predict(customer.to_dataframe())[0]
        label    = "Will Churn" if prob >= THRESHOLD else "Will Stay"

        st.divider()
        st.header("Result")

        col_a, col_b = st.columns(2)
        col_a.metric("Prediction", label)
        col_b.metric("Churn Probability", f"{prob:.1%}")

        st.progress(float(prob))

        if prob >= THRESHOLD:
            st.error(
                f"High churn risk (probability {prob:.1%} exceeds "
                f"threshold {THRESHOLD:.0%}) — consider a retention offer."
            )
        elif prob >= 0.2:
            st.warning(
                f"Moderate churn risk ({prob:.1%}) — worth monitoring."
            )
        else:
            st.success(
                f"Low churn risk ({prob:.1%}) — customer likely to stay."
            )

        # Show what drove the prediction
        st.divider()
        st.subheader("Key risk factors present")
        factors = []
        if contract == "Month-to-month":
            factors.append("Month-to-month contract — highest churn risk contract type")
        if internet == "Fiber optic":
            factors.append("Fiber optic internet — associated with higher churn in training data")
        if payment == "Electronic check":
            factors.append("Electronic check payment — less committed payment method")
        if tenure <= 12:
            factors.append(f"Short tenure ({tenure} months) — new customers churn more")
        if senior_citizen == "Yes":
            factors.append("Senior citizen — slightly higher churn rate")
        if not factors:
            factors.append("No major risk factors detected")
        for f in factors:
            st.write(f"- {f}")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")