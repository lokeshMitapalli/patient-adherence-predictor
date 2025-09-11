import streamlit as st
import pandas as pd
import os
from joblib import load
import pickle
from io import BytesIO

st.title("Patient Adherence Prediction Dashboard")

# === HELPERS ===
def show_toast(message, color="green"):
    toast_html = f"""
    <div style="
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: {color};
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        font-size: 16px;
        z-index: 9999;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    ">{message}</div>
    <script>
        setTimeout(function(){{
            var toasts = document.querySelectorAll('[style*="position: fixed; bottom: 20px;"]');
            toasts.forEach(function(toast){{ toast.style.display = 'none'; }});
        }}, 3000);
    </script>
    """
    st.markdown(toast_html, unsafe_allow_html=True)

def encode_dataframe(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes
    return df

def check_missing_dosage(df):
    follow_up = df.get("Follow_Up_Days", pd.Series([None]*len(df)))
    dosage = df.get("Dosage_mg", pd.Series([0]*len(df)))
    alerts = df[(df["Predicted_Adherence"] == "Non-Adherent") |
                (dosage == 0) | (follow_up.isna())]
    return alerts

def load_model(path_or_file):
    try:
        # If path given
        if isinstance(path_or_file, (str, os.PathLike)):
            return load(path_or_file)
        # If uploaded file (BytesIO)
        else:
            content = path_or_file.read()
            return pickle.loads(content)
    except:
        if isinstance(path_or_file, (str, os.PathLike)):
            with open(path_or_file, "rb") as f:
                return pickle.load(f)
        else:
            path_or_file.seek(0)
            return pickle.load(path_or_file)

def ensure_features(df, model):
    trained_features = model.feature_names_in_
    for col in trained_features:
        if col not in df.columns:
            df[col] = 0
    return df[trained_features]

# === MODEL LOADING ===
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = None

if os.path.exists(model_path):
    st.info("Loading model from project folder...")
    try:
        model = load_model(model_path)
        show_toast("✅ Model loaded successfully!", "green")
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.warning("No model found. Please upload one.")
    uploaded_model = st.file_uploader("Upload your model file (.pkl)", type=["pkl"])
    if uploaded_model:
        try:
            model = load_model(uploaded_model)
            show_toast("✅ Model uploaded successfully!", "green")
        except Exception as e:
            st.error(f"Error loading uploaded model: {e}")

if model is None:
    st.stop()

# === DATASET LOADING ===
st.sidebar.header("Upload Your Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    show_toast("✅ Dataset uploaded successfully!", "green")
else:
    default_path = os.path.join(os.path.dirname(__file__), "patient_adherence_dataset.csv")
    if os.path.exists(default_path):
        data = pd.read_csv(default_path)
        st.info("Using default dataset from project folder")
    else:
        st.error("No dataset found. Please upload one to continue.")
        st.stop()

st.write("### Dataset Preview")
st.dataframe(data.head())

# === TARGET / FEATURES ===
y = None
if "Adherence" in data.columns:
    y = data["Adherence"].fillna("").apply(lambda x: 1 if str(x).strip().lower() == "adherent" else 0)
X = data.drop(columns=["Adherence"], errors="ignore")

# === SINGLE PREDICTION ===
st.sidebar.header("Make a Single Prediction")
input_data = {col: st.sidebar.text_input(f"Enter {col}") for col in X.columns}

if st.sidebar.button("Predict"):
    try:
        input_df = pd.DataFrame([input_data])
        input_df = encode_dataframe(input_df)

        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="ignore")

        input_df = ensure_features(input_df, model)
        prediction = model.predict(input_df)[0]
        result = "Adherent" if prediction == 1 else "Non-Adherent"
        show_toast(f"✅ Single prediction: {result}", "green")
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# === BATCH PREDICTION ===
st.subheader("Batch Prediction on Dataset")
if st.button("Run Batch Prediction"):
    try:
        X_copy = encode_dataframe(X.copy())
        X_copy = ensure_features(X_copy, model)

        preds = model.predict(X_copy)
        data["Predicted_Adherence"] = ["Adherent" if p == 1 else "Non-Adherent" for p in preds]

        st.write("### Full Dataset with Predictions")
        st.dataframe(data)

        alerts = check_missing_dosage(data)
        if not alerts.empty:
            st.warning(f"{len(alerts)} patients missing doses or non-adherent!")
            st.dataframe(alerts)
        else:
            st.success("All patients are adherent and up-to-date on dosages!")

        buffer = BytesIO()
        data.to_csv(buffer, index=False)
        st.download_button(
            "Download Predictions as CSV",
            buffer.getvalue(),
            "patient_predictions.csv",
            "text/csv"
        )
    except Exception as e:
        st.error(f"Error during batch prediction: {e}")



