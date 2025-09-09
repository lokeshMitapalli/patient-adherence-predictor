import streamlit as st
import pandas as pd
import os
from joblib import load
import pickle
from io import BytesIO

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

X = data.drop(columns=["Adherence"], errors='ignore')

# === HELPER: ENCODE CATEGORICALS ===
def encode_dataframe(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
    return df

# === SINGLE PREDICTION ===
st.sidebar.header("Make a Single Prediction")
input_data = {}
for col in X.columns:
    value = st.sidebar.text_input(f"Enter {col}")
    input_data[col] = value

if st.sidebar.button("Predict"):
    try:
        input_df = pd.DataFrame([input_data])

        # Encode categorical values
        input_df = encode_dataframe(input_df)

        # Convert numeric fields
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except:
                pass

        # Align columns with model
        trained_features = model.feature_names_in_
        for col in trained_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[trained_features]

        prediction = model.predict(input_df)[0]
        result = "Adherent" if prediction == 1 else "Non-Adherent"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# === BATCH PREDICTION ===
st.subheader("Batch Prediction on Uploaded Dataset")
if st.button("Run Batch Prediction"):
    try:
        X_copy = X.copy()

        # Encode categorical columns automatically
        X_copy = encode_dataframe(X_copy)

        # Align columns with model
        trained_features = model.feature_names_in_
        for col in trained_features:
            if col not in X_copy.columns:
                X_copy[col] = 0
        X_copy = X_copy[trained_features]

        preds = model.predict(X_copy)
        data["Predicted_Adherence"] = ["Adherent" if p == 1 else "Non-Adherent" for p in preds]
        st.write("### Dataset with Predictions")
        st.dataframe(data.head())

        # Prepare file for download
        buffer = BytesIO()
        data.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="Download Predictions as CSV",
            data=buffer,
            file_name="patient_predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error during batch prediction: {e}")






