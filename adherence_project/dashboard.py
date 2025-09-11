import streamlit as st
import pandas as pd
import os
from joblib import load
from io import BytesIO

st.title("Patient Adherence Prediction Dashboard")

# === HELPER: Toast Notifications ===
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
    ">
        {message}
    </div>
    <script>
        setTimeout(function(){{
            var toasts = document.querySelectorAll('[style*="position: fixed; bottom: 20px;"]');
            toasts.forEach(function(toast){{ toast.style.display = 'none'; }});
        }}, 3000);
    </script>
    """
    st.markdown(toast_html, unsafe_allow_html=True)

# === HELPER: Encode categorical columns ===
def encode_dataframe(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
    return df

# === HELPER: Check missing dosages / non-adherence ===
def check_missing_dosage(df):
    follow_up = df["Follow_Up_Days"] if "Follow_Up_Days" in df.columns else pd.Series([None]*len(df))
    dosage = df["Dosage_mg"] if "Dosage_mg" in df.columns else pd.Series([0]*len(df))
    alerts = df[(df["Predicted_Adherence"] == "Non-Adherent") |
                (dosage == 0) |
                (follow_up.isna())]
    return alerts

# === MODEL LOADING ===
model = None
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

if os.path.exists(model_path):
    st.info("Loading model from project folder...")
    try:
        model = load(model_path)
        model_name = model.__class__.__name__
        show_toast(f"✅ Model loaded successfully! ({model_name})", "green")
        st.success(f"Model loaded: **{model_name}** ✅")
    except Exception as e:
        show_toast("❌ Error loading model!", "red")
        st.error(f"Error loading model: {e}")
else:
    st.warning("No model found in project folder. Please upload one.")
    uploaded_model = st.file_uploader("Upload your model file (.pkl)", type=["pkl"])
    if uploaded_model:
        try:
            temp_path = "uploaded_model.pkl"
            with open(temp_path, "wb") as f:
                f.write(uploaded_model.read())
            model = load(temp_path)
            model_name = model.__class__.__name__
            show_toast(f"✅ Uploaded model loaded! ({model_name})", "green")
            st.success(f"Model loaded: **{model_name}** ✅")
        except Exception as e:
            show_toast("❌ Error loading uploaded model!", "red")
            st.error(f"Error loading uploaded model: {e}")

if model is None:
    st.stop()

# === DATASET UPLOAD ===
st.sidebar.header("Upload Your Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    show_toast("✅ Dataset uploaded successfully!", "green")
    st.write("### Uploaded Dataset Preview")
    st.dataframe(data.head())
else:
    default_path = os.path.join(os.path.dirname(__file__), 'patient_adherence_dataset.csv')
    if os.path.exists(default_path):
        data = pd.read_csv(default_path)
        st.write("### Default Dataset Preview")
        st.dataframe(data.head())
    else:
        show_toast("❌ No dataset found!", "red")
        st.error("No dataset found. Please upload a CSV file to continue.")
        st.stop()

# === PROCESS DATA ===
if "Adherence" in data.columns:
    y = data["Adherence"].fillna("").apply(lambda x: 1 if str(x).strip().lower() == "adherent" else 0)
else:
    st.warning("The dataset does not contain an 'Adherence' column. Predictions will be based on features only.")
    y = None

X = data.drop(columns=["Adherence"], errors='ignore')

# === SINGLE PREDICTION ===
st.sidebar.header("Make a Single Prediction")
input_data = {}
for col in X.columns:
    value = st.sidebar.text_input(f"Enter {col}")
    input_data[col] = value

if st.sidebar.button("Predict"):
    try:
        input_df = pd.DataFrame([input_data])
        input_df = encode_dataframe(input_df)
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except:
                pass
        trained_features = model.feature_names_in_
        for col in trained_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[trained_features]
        prediction = model.predict(input_df)[0]
        result = "Adherent" if prediction == 1 else "Non-Adherent"
        show_toast(f"✅ Single prediction: {result}", "green")
        st.success(f"Prediction: {result}")
    except Exception as e:
        show_toast("❌ Error during single prediction!", "red")
        st.error(f"Error during prediction: {e}")

# === BATCH PREDICTION ===
st.subheader("Batch Prediction on Uploaded Dataset")
if st.button("Run Batch Prediction"):
    try:
        show_toast("Running batch prediction...", "#007bff")
        X_copy = X.copy()
        X_copy = encode_dataframe(X_copy)
        trained_features = model.feature_names_in_
        for col in trained_features:
            if col not in X_copy.columns:
                X_copy[col] = 0
        X_copy = X_copy[trained_features]
        preds = model.predict(X_copy)
        data["Predicted_Adherence"] = ["Adherent" if p == 1 else "Non-Adherent" for p in preds]
        show_toast("✅ Batch prediction completed successfully!", "green")
        st.write("### Full Dataset with Predictions")
        st.dataframe(data)
        alerts = check_missing_dosage(data)
        if not alerts.empty:
            show_toast(f"⚠ {len(alerts)} patients missing doses or non-adherent!", "orange")
            st.warning("### Patients Needing Attention")
            st.dataframe(alerts)
        else:
            show_toast("✅ All patients are adherent!", "green")
            st.success("All patients are adherent and up-to-date on dosages!")
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
        show_toast("❌ Error during batch prediction!", "red")
        st.error(f"Error during batch prediction: {e}")





