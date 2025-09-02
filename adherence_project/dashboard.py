import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("Medication Adherence Predictor")
st.markdown("This app predicts if a patient will adhere to their medication using a model trained on synthetic data.")

# ✅ Generate synthetic dataset
@st.cache_data
def create_data(n=500):
    np.random.seed(42)
    genders = ["Male", "Female", "Unknown"]
    medications = ["Type1", "Type2"]

    data = pd.DataFrame({
        "Gender": np.random.choice(genders, n),
        "Age": np.random.randint(18, 80, n),
        "Medication_Type": np.random.choice(medications, n),
        "Missed_Doses": np.random.randint(0, 10, n),
        "Days_Since_Last_Visit": np.random.randint(0, 60, n),
        "App_Usage": np.random.choice(["Yes", "No"], n),
        "Adherence": np.random.choice([1, 0], n)  # 1 = Adherent, 0 = Non-Adherent
    })
    return data

data = create_data()

# ✅ Train model at runtime
def train_model(data):
    X = data[["Gender", "Age", "Medication_Type", "Missed_Doses", "Days_Since_Last_Visit", "App_Usage"]]
    y = data["Adherence"]

    X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns

model, feature_cols = train_model(data)

# ✅ User Input Form
st.write("### Enter Patient Details")
gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
age = st.number_input("Age", min_value=0, max_value=120)
medication_type = st.selectbox("Medication Type", ["Type1", "Type2"])
missed_doses = st.number_input("Missed Doses", min_value=0)
last_visit_gap = st.number_input("Days since Last Visit", min_value=0)
app_usage = st.selectbox("App Usage", ["Yes", "No"])

# ✅ Prepare input for prediction
input_data = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "Medication_Type": [medication_type],
    "Missed_Doses": [missed_doses],
    "Days_Since_Last_Visit": [last_visit_gap],
    "App_Usage": [app_usage]
})

input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=feature_cols, fill_value=0)

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success("Prediction: Adherent" if prediction == 1 else "Prediction: Non-Adherent")
