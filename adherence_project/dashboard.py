import streamlit as st
import pandas as pd
import joblib
import os

st.title("Medication Adherence Predictor")
st.markdown("This app predicts if a patient will adhere to their medication.")

# Load model safely
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found! Please ensure model.pkl is in the same folder as this script.")
    st.stop()

# Input fields
st.write("Enter Patient Details")
gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
age = st.number_input("Age", min_value=0, max_value=120)
medication_type = st.selectbox("Medication Type", ["Type1", "Type2"])
missed_doses = st.number_input("Missed Doses", min_value=0)
last_visit_gap = st.number_input("Days since Last Visit", min_value=0)
app_usage = st.selectbox("App Usage", ["Yes", "No"])

# Encoding
gender_encoded = 0 if gender == "Male" else (1 if gender == "Female" else 2)
medication_encoded = 0 if medication_type == "Type1" else 1
app_usage_encoded = 1 if app_usage == "Yes" else 0

# Features
features = [[gender_encoded, age, medication_encoded, missed_doses, last_visit_gap, app_usage_encoded]]

# Predict
if st.button("Predict"):
    prediction = model.predict(features)[0]
    st.success("Adherent ✅" if prediction == 1 else "Non-Adherent ❌")
