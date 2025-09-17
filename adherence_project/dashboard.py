import streamlit as st
import pandas as pd
import os
from joblib import load
from io import BytesIO
import smtplib
from email.mime.text import MIMEText

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

# === HELPER: Send Email Alert ===
def send_email_alert(patient_id, recipient_email):
    msg = MIMEText(f"⚠ Alert: Patient {patient_id} is NON-ADHERENT. Please follow up immediately.")
    msg["Subject"] = "🚨 Non-Adherence Alert"
    msg["From"] = "your_email@gmail.com"  # Replace with your Gmail
    msg["To"] = recipient_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login("mittapallilokeswarreddy10@gmail.com", "loki10042005")  # Use App Password
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email sending failed: {e}")
        return False

# === MODEL LOADING (ALWAYS LOCAL) ===
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
if os.path.exists(model_path):
    st.info("Loading model from project folder...")
    try:
        model = load(model_path)
        show_toast("✅ Model loaded successfully!", color="green")
    except Exception as e:
        show_toast("❌ Error loading model!", color="red")
        st.error(f"Error loading model: {e}")
        st.stop()
else:
    st.error("❌ model.pkl not found in project folder! Please add it.")
    st.stop()

# === DATASET UPLOAD ===
st.sidebar.header("Upload Your Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    show_toast("✅ Dataset uploaded successfully!", color="green")
    st.write("### Uploaded Dataset Preview")
    st.dataframe(data.head())
else:
    default_path = os.path.join(os.path.dirname(__file__), "patient_adherence_dataset.csv")
    if os.path.exists(default_path):
        data = pd.read_csv(default_path)
        st.write("### Default Dataset Preview")
        st.dataframe(data.head())
    else:
        st.error("❌ No dataset found. Please upload a CSV file to continue.")
        st.stop()

# === PROCESS DATA ===
if "Adherence" in data.columns:
    y = data["Adherence"].fillna("").apply(lambda x: 1 if str(x).strip().lower() == "adherent" else 0)
else:
    y = None
    st.warning("The dataset does not contain an 'Adherence' column. Predictions will be based on features only.")

X = data.drop(columns=["Adherence"], errors="ignore")

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
        show_toast(f"✅ Single prediction: {result}", color="green")
        st.success(f"Prediction: {result}")
    except Exception as e:
        show_toast("❌ Error during single prediction!", color="red")
        st.error(f"Error during prediction: {e}")

# === BATCH PREDICTION ===
st.subheader("Batch Prediction on Dataset")
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

        # 📊 Adherence Overview
        if "Predicted_Adherence" in data.columns:
            st.subheader("📊 Adherence Overview")

            adherence_counts = data["Predicted_Adherence"].value_counts()
            st.bar_chart(adherence_counts)

            # ⚠ Non-Adherent Alerts
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
                st.success("All patients are adherent and up-to-date on dosages!")

        # Download predictions
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

