import streamlit as st
import pandas as pd
import pickle
from io import BytesIO
import base64
import smtplib
from email.mime.text import MIMEText
import os

st.title("Patient Adherence Prediction Dashboard")

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
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
    return df

def send_email_alert(patient_id, recipient_email):
    msg = MIMEText(f"⚠ Alert: Patient {patient_id} is NON-ADHERENT. Please follow up immediately.")
    msg["Subject"] = "🚨 Non-Adherence Alert"
    msg["From"] = "your_email@gmail.com"
    msg["To"] = recipient_email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login("mittapallilokeswarreddy10@gmail.com", "loki10042005")
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email sending failed: {e}")
        return False

# ----------------- EMBEDDED MODEL WITH VALIDATION -----------------
model_base64 = """
<PASTE-YOUR-REAL-MODEL-BASE64-HERE>
"""

model = None
try:
    if not model_base64.strip():
        raise ValueError("The embedded model base64 string is empty.")
    try:
        model = pickle.loads(base64.b64decode(model_base64))
        if not hasattr(model, "predict") or not hasattr(model, "feature_names_in_"):
            raise ValueError("The decoded object is not a valid trained model.")
        show_toast("✅ Embedded model loaded successfully!", color="green")
    except Exception as inner:
        raise ValueError(f"Failed to decode or load the embedded model: {inner}")
except Exception as e:
    st.error(f"❌ Embedded model could not be loaded.\nDetails: {e}\nPlease make sure the base64 string is correct.")
    st.stop()

# ----------------- DATASET UPLOAD -----------------
st.sidebar.header("Upload Your Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        show_toast("✅ Dataset uploaded successfully!", color="green")
        st.write("### Uploaded Dataset Preview")
        st.dataframe(data.head())
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()
else:
    default_path = 'patient_adherence_dataset.csv'
    if os.path.exists(default_path):
        data = pd.read_csv(default_path)
        st.write("### Default Dataset Preview")
        st.dataframe(data.head())
    else:
        data = pd.DataFrame(columns=model.feature_names_in_)
        st.write("### Using empty dataset template")
        st.dataframe(data.head())

if "Adherence" in data.columns:
    y = data["Adherence"].fillna("").apply(lambda x: 1 if str(x).strip().lower() == "adherent" else 0)
else:
    y = None

X = data.drop(columns=["Adherence"], errors='ignore')

# ----------------- SINGLE PREDICTION -----------------
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
        show_toast(f"✅ Single prediction: {result}", color="green")
        st.success(f"Prediction: {result}")
    except Exception as e:
        show_toast("❌ Error during single prediction!", color="red")
        st.error(f"Error during prediction: {e}")

# ----------------- BATCH PREDICTION -----------------
st.subheader("Batch Prediction on Uploaded Dataset")
if st.button("Run Batch Prediction"):
    try:
        show_toast("Running batch prediction...", color="#007bff")
        X_copy = X.copy()
        X_copy = encode_dataframe(X_copy)
        trained_features = model.feature_names_in_
        for col in trained_features:
            if col not in X_copy.columns:
                X_copy[col] = 0
        X_copy = X_copy[trained_features]
        preds = model.predict(X_copy)
        data["Predicted_Adherence"] = ["Adherent" if p == 1 else "Non-Adherent" for p in preds]
        show_toast("✅ Batch prediction completed successfully!", color="green")
        st.write("### Full Dataset with Predictions")
        st.dataframe(data)
        adherence_counts = data["Predicted_Adherence"].value_counts()
        st.subheader("📊 Adherence Overview")
        st.bar_chart(adherence_counts)
        ratio_df = (adherence_counts / adherence_counts.sum() * 100).reset_index()
        ratio_df.columns = ["Adherence_Status", "Percentage"]
        st.dataframe(ratio_df)
        st.area_chart(ratio_df.set_index("Adherence_Status"))
        non_adherent = data[data["Predicted_Adherence"] == "Non-Adherent"]
        if not non_adherent.empty:
            st.error(f"⚠ {len(non_adherent)} NON-ADHERENT patients found!")
            st.dataframe(non_adherent)
            recipient = st.text_input("Doctor Email", "doctor@example.com")
            if st.button("Send Email Alerts"):
                for _, row in non_adherent.iterrows():
                    patient_id = row.get("Patient_ID", "Unknown")
                    send_email_alert(patient_id, recipient)
                st.success("✅ Email alerts sent successfully!")
        else:
            st.success("All patients are adherent!")
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
        show_toast("❌ Error during batch prediction!", color="red")
        st.error(f"Error during batch prediction: {e}")










