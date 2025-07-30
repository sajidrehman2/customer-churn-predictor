import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("random_forest_model.pkl")

# Streamlit UI
st.title("Churn Prediction")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", value=70.0)
total_charges = st.number_input("Total Charges", value=3000.0)

# Encode all inputs manually or use same pipeline from training
input_dict = {
    'gender': 1 if gender == "Male" else 0,
    'SeniorCitizen': 1 if senior == "Yes" else 0,
    'Partner': 1 if partner == "Yes" else 0,
    'Dependents': 1 if dependents == "Yes" else 0,
    'tenure': tenure,
    'PhoneService': 1 if phone_service == "Yes" else 0,
    'MultipleLines': 1 if multiple_lines == "Yes" else 0,
    'InternetService': {"DSL": 0, "Fiber optic": 1, "No": 2}[internet_service],
    'OnlineSecurity': {"Yes": 1, "No": 0, "No internet service": 2}[online_security],
    'OnlineBackup': {"Yes": 1, "No": 0, "No internet service": 2}[online_backup],
    'DeviceProtection': {"Yes": 1, "No": 0, "No internet service": 2}[device_protection],
    'TechSupport': {"Yes": 1, "No": 0, "No internet service": 2}[tech_support],
    'StreamingTV': {"Yes": 1, "No": 0, "No internet service": 2}[streaming_tv],
    'StreamingMovies': {"Yes": 1, "No": 0, "No internet service": 2}[streaming_movies],
    'Contract': {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
    'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
    'PaymentMethod': {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }[payment_method],
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

input_df = pd.DataFrame([input_dict])

if st.button("Predict Churn"):
    result = model.predict(input_df)
    if result[0] == 1:
        st.error("⚠️ This customer is likely to CHURN.")
    else:
        st.success("✅ This customer is NOT likely to churn.")
