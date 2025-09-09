import streamlit as st
import pandas as pd
import os
from joblib import load

st.title("Patient Adherence Prediction Dashboard")

# Load the trained model safely
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

if not os.path.exists(model_path):
    st.error("Model file 'model.pkl' not found. Please upload it to the project folder and restart the app.")
    st.stop()

try:
    model = load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Option to upload dataset
st.sidebar.header("Upload Your Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset Preview")
    st.dataframe(data.head())
else:
    default_path = os.path.join(os.path.dirname(__file__), 'patient_adherence_dataset.csv')
    if os.path.exists(default_path):
        data = pd.read_csv(default_path)
        st.write("### Default Dataset Preview")
        st.dataframe(data.head())
    else:
        st.error("Default dataset file not found. Please upload a dataset.")
        st.stop()

# Check if 'Adherence' column exists
if "Adherence" in data.columns:
    y = data["Adherence"].fillna("").apply(lambda x: 1 if str(x).strip().lower() == "adherent" else 0)
else:
    st.warning("The dataset does not contain an 'Adherence' column. Predictions will be based on features only.")
    y = None

# Get feature columns (exclude Adherence if present)
X = data.drop(columns=["Adherence"], errors='ignore')

# Sidebar inputs for prediction
st.sidebar.header("Make a Prediction")
input_data = {}
for col in X.columns:
    value = st.sidebar.text_input(f"Enter {col}")
    input_data[col] = value

if st.sidebar.button("Predict"):
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Convert columns to numeric where possible
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except:
                pass

        prediction = model.predict(input_df)[0]
        result = "Adherent" if prediction == 1 else "Non-Adherent"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")


