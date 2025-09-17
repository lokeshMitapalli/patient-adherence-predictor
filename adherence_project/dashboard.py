import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from io import BytesIO

st.title("📊 Patient Adherence Prediction Dashboard")

# === PATH FOR MODEL ===
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = None

# === TRY TO LOAD MODEL ===
if os.path.exists(model_path):
    try:
        st.info("✅ Loading pre-trained model...")
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"⚠ Model file is corrupted: {e}")
        model = None

# === TRAIN IF NO MODEL ===
if model is None:
    st.warning("⚠ No valid model found. Training a new one from dataset...")

    dataset_path = os.path.join(os.path.dirname(__file__), "patient_adherence_dataset.csv")
    if os.path.exists(dataset_path):
        data = pd.read_csv(dataset_path)

        if "Adherence" not in data.columns:
            st.error("Dataset must contain an 'Adherence' column!")
            st.stop()

        X = data.drop(columns=["Adherence"], errors="ignore")
        y = data["Adherence"].apply(lambda x: 1 if str(x).strip().lower() == "adherent" else 0)

        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype("category").cat.codes

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        joblib.dump(model, model_path)

        acc = accuracy_score(y_test, model.predict(X_test))
        st.success(f"✅ Model trained successfully! Accuracy: {acc:.2f}")
    else:
        st.error("❌ No dataset found! Please upload patient_adherence_dataset.csv.")
        st.stop()

# === SINGLE PREDICTION ===
st.subheader("🔮 Single Patient Prediction")
patient_input = {}
X_columns = model.feature_names_in_

for col in X_columns:
    patient_input[col] = st.text_input(f"Enter {col}")

if st.button("Predict"):
    input_df = pd.DataFrame([patient_input])
    for col in input_df.select_dtypes(include="object").columns:
        input_df[col] = input_df[col].astype("category").cat.codes

    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)[0]
    result = "Adherent ✅" if prediction == 1 else "Non-Adherent ❌"
    st.success(f"Prediction: {result}")

# === BATCH PREDICTION ===
st.subheader("📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", batch_data.head())

    for col in batch_data.select_dtypes(include="object").columns:
        batch_data[col] = batch_data[col].astype("category").cat.codes

    for col in model.feature_names_in_:
        if col not in batch_data.columns:
            batch_data[col] = 0
    batch_data = batch_data[model.feature_names_in_]

    preds = model.predict(batch_data)
    batch_data["Predicted_Adherence"] = ["Adherent" if p == 1 else "Non-Adherent" for p in preds]

    st.write("### Predictions", batch_data)

    # === 📊 Graphs & Ratios ===
    st.subheader("📊 Adherence Overview")
    adherence_counts = batch_data["Predicted_Adherence"].value_counts()
    st.bar_chart(adherence_counts)

    ratio_df = (adherence_counts / adherence_counts.sum() * 100).reset_index()
    ratio_df.columns = ["Adherence_Status", "Percentage"]
    st.write("### Adherence Ratios (%)")
    st.dataframe(ratio_df)

    # === ⚠️ Alerts ===
    non_adherent = batch_data[batch_data["Predicted_Adherence"] == "Non-Adherent"]
    if not non_adherent.empty:
        st.error(f"⚠ {len(non_adherent)} NON-ADHERENT patients found!")
        st.dataframe(non_adherent)
    else:
        st.success("🎉 All patients are adherent!")

    # === Download results ===
    buffer = BytesIO()
    batch_data.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label="⬇ Download Predictions as CSV",
        data=buffer,
        file_name="patient_predictions.csv",
        mime="text/csv"
    )
te("### Predictions", batch_data)
    st.download_button("Download Predictions", batch_data.to_csv(index=False), "predictions.csv", "text/csv")



