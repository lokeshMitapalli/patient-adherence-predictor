import streamlit as st
import pandas as pd
import os
from joblib import load
import pickle

st.title("Patient Adherence Prediction Dashboard")

# === MODEL LOADING ===
model = None
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

if os.path.exists(model_path):
    st.info("Loading model from project folder...")
    try:
        model = load(model_path)
    except:
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            model = None

if model is None:
    st.warning("No valid model found in project folder. Please upload your model file.")
    uploaded_model = st.file_uploader("Upload your model file (.pkl)", type=["pkl"])
    if uploaded_model:
        try:
            model = load(uploaded_model)
        except:
            try:
                model = pickle.load(uploaded_model)
            except Exception as e:
                st.error(f"Error loading uploaded model: {e}")
                model = None

if model is None:
    st.stop()

# === DATASET LOADING ===
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
        st.error("No dataset found. Please upload a CSV file to continue.")
        st.stop()

# === PROCESS DATA ===
if "Adherence" in data.columns:
    y = data["Adherence"].fillna("").apply(lambda x: 1 if str(x).strip().lower() == "adherent" else 0)
else:
    st.warning("The dataset does not contain an 'Adherence' column. Predictions will be based on features only.")
    y = None

# Features
X = data.drop(columns=["Adherence"], errors='ignore')

# === PREDICTION UI ===
st.sidebar.header("Make a Prediction")
input_data = {}
for col in X.columns:
    value = st.sidebar.text_input(f"Enter {col}")
    input_data[col] = value

if st.sidebar.button("Predict"):
    try:
        input_df = pd.DataFrame([input_data])

        # Convert numeric fields
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



