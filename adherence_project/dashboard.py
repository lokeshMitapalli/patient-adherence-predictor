import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from io import BytesIO
import numpy as np

st.title("📊 Patient Adherence Prediction Dashboard")

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = None

# Load pre-trained model if exists
if os.path.exists(model_path):
    try:
        st.info("✅ Loading pre-trained model...")
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"⚠ Model file is corrupted: {e}")
        model = None

# Train new model if none found
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

        # Encode categorical columns
        category_mappings = {}
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype("category")
            category_mappings[col] = {cat: code for code, cat in enumerate(X[col].cat.categories)}
            X[col] = X[col].cat.codes

        # Stratified split to preserve class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        # Train model
        model = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)
        model.fit(X_train, y_train)

        # Save model and mappings
        joblib.dump({"model": model, "mappings": category_mappings}, model_path)

        acc = accuracy_score(y_test, model.predict(X_test))
        st.success(f"✅ Model trained successfully! Accuracy: {acc:.2f}")
    else:
        st.error("❌ No dataset found! Please upload patient_adherence_dataset.csv.")
        st.stop()
else:
    # Load category mappings from model file
    loaded = joblib.load(model_path)
    model = loaded["model"]
    category_mappings = loaded.get("mappings", {})

# --- Single Patient Prediction ---
st.subheader("🔮 Single Patient Prediction")
patient_input = {}
X_columns = model.feature_names_in_

for col in X_columns:
    patient_input[col] = st.text_input(f"Enter {col}")

if st.button("Predict"):
    input_df = pd.DataFrame([patient_input])
    # Apply saved category mappings
    for col, mapping in category_mappings.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].map(mapping).fillna(0)

    # Ensure all model features are present
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)[0]
    result = "Adherent ✅" if prediction == 1 else "Non-Adherent ❌"
    st.success(f"Prediction: {result}")

# --- Batch Prediction ---
st.subheader("📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", batch_data.head())

    # Apply category mappings
    for col, mapping in category_mappings.items():
        if col in batch_data.columns:
            batch_data[col] = batch_data[col].map(mapping).fillna(0)

    # Ensure all model features are present
    for col in model.feature_names_in_:
        if col not in batch_data.columns:
            batch_data[col] = 0
    batch_data = batch_data[model.feature_names_in_]

    preds = model.predict(batch_data)
    batch_data["Predicted_Adherence"] = ["Adherent" if p == 1 else "Non-Adherent" for p in preds]

    st.write("### Predictions", batch_data)

    # Adherence Overview
    st.subheader("📊 Adherence Overview")
    adherence_counts = batch_data["Predicted_Adherence"].value_counts()
    st.bar_chart(adherence_counts)

    ratio_df = (adherence_counts / adherence_counts.sum() * 100).reset_index()
    ratio_df.columns = ["Adherence_Status", "Percentage"]
    st.write("### Adherence Ratios (%)")
    st.dataframe(ratio_df)

    # Highlight non-adherent patients
    non_adherent = batch_data[batch_data["Predicted_Adherence"] == "Non-Adherent"]
    if not non_adherent.empty:
        st.error(f"⚠ {len(non_adherent)} NON-ADHERENT patients found!")
        st.dataframe(non_adherent)
    else:
        st.success("🎉 All patients are adherent!")

    # Download predictions
    buffer = BytesIO()
    batch_data.to_csv(buffer, index=False)
    buffer.seek(0)

    st.download_button(
        label="⬇ Download Predictions as CSV",
        data=buffer,
        file_name="patient_predictions.csv",
        mime="text/csv"
    )





