import streamlit as st
import pandas as pd
import os
from joblib import load
import pickle
from io import BytesIO
import plotly.express as px  # ✅ for pie charts

st.set_page_config(page_title="Patient Adherence Dashboard", layout="wide")
st.title("🩺 Patient Adherence Prediction Dashboard")

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
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

if os.path.exists(model_path):
    st.info("Loading model from project folder...")
    try:
        model = load(model_path)
        show_toast("✅ Model loaded successfully!", color="green")
    except:
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                show_toast("✅ Model loaded successfully!", color="green")
        except Exception as e:
            show_toast("❌ Error loading model!", color="red")
            st.error(f"Error loading model: {e}")
            model = None

if model is None:
    st.warning("No valid model found in project folder. Please upload your model file.")
    uploaded_model = st.file_uploader("Upload your model file (.pkl)", type=["pkl"])
    if uploaded_model:
        try:
            model = load(uploaded_model)
            show_toast("✅ Model uploaded successfully!", color="green")
        except:
            try:
                model = pickle.load(uploaded_model)
                show_toast("✅ Model uploaded successfully!", color="green")
            except Exception as e:
                show_toast("❌ Error loading uploaded model!", color="red")
                st.error(f"Error loading uploaded model: {e}")
                model = None

if model is None:
    st.stop()

# === DATASET UPLOAD ===
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    show_toast("✅ Dataset uploaded successfully!", color="green")
    st.write("### Uploaded Dataset Preview")
    st.dataframe(data.head())
else:
    default_path = os.path.join(os.path.dirname(__file__), 'patient_adherence_dataset.csv')
    if os.path.exists(default_path):
        data = pd.read_csv(default_path)
        st.write("### Default Dataset Preview")
        st.dataframe(data.head())
    else:
        show_toast("❌ No dataset found!", color="red")
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
st.sidebar.header("🔮 Make a Single Prediction")
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
st.subheader("📊 Batch Prediction on Dataset")
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

        # === KPIs ===
        st.markdown("### 📌 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)

        total_patients = len(data)
        adherent_count = (data["Predicted_Adherence"] == "Adherent").sum()
        non_adherent_count = (data["Predicted_Adherence"] == "Non-Adherent").sum()
        alerts = check_missing_dosage(data)
        alerts_count = len(alerts)

        col1.metric("Total Patients", total_patients)
        col2.metric("Adherent", adherent_count)
        col3.metric("Non-Adherent", non_adherent_count)
        col4.metric("⚠ At-Risk", alerts_count)

        # === Graphs ===
        st.subheader("📈 Adherence Insights")
        adherence_counts = data["Predicted_Adherence"].value_counts().reset_index()
        adherence_counts.columns = ["Adherence", "Count"]

        # Bar chart
        st.bar_chart(adherence_counts.set_index("Adherence"))

        # Pie chart with Plotly
        fig = px.pie(adherence_counts, names="Adherence", values="Count",
                     color="Adherence", color_discrete_map={"Adherent": "green", "Non-Adherent": "red"})
        st.plotly_chart(fig, use_container_width=True)

        # Dosage & Follow-Up Trends
        if "Dosage_mg" in data.columns:
            st.subheader("💊 Average Dosage by Adherence")
            avg_dosage = data.groupby("Predicted_Adherence")["Dosage_mg"].mean().reset_index()
            st.bar_chart(avg_dosage.set_index("Predicted_Adherence"))

        if "Follow_Up_Days" in data.columns:
            st.subheader("📅 Average Follow-Up Days by Adherence")
            avg_followup = data.groupby("Predicted_Adherence")["Follow_Up_Days"].mean().reset_index()
            st.bar_chart(avg_followup.set_index("Predicted_Adherence"))

        # === Alerts ===
        if not alerts.empty:
            show_toast(f"⚠ {len(alerts)} patients missing doses or non-adherent!", color="orange")
            st.warning("### ⚠ Patients Needing Attention")
            st.dataframe(alerts)

            # Risk Pie with Plotly
            st.subheader("⚠ Risk Overview")
            risk_data = pd.DataFrame({"Status": ["At-Risk", "Safe"], 
                                      "Count": [alerts_count, total_patients - alerts_count]})
            fig2 = px.pie(risk_data, names="Status", values="Count",
                          color="Status", color_discrete_map={"At-Risk": "orange", "Safe": "green"})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            show_toast("✅ All patients are adherent!", color="green")
            st.success("All patients are adherent and up-to-date on dosages!")

        # === Download button ===
        buffer = BytesIO()
        data.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="💾 Download Predictions as CSV",
            data=buffer,
            file_name="patient_predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        show_toast("❌ Error during batch prediction!", color="red")
        st.error(f"Error during batch prediction: {e}")
