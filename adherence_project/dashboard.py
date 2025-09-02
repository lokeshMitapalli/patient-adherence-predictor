import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("Medication Adherence Predictor")
st.markdown("This app predicts if a patient will adhere to their medication.")

# Load dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("patient_adherence_dataset.csv")
        return data
    except FileNotFoundError:
        st.error("Dataset not found! Please upload patient_adherence_dataset.csv in the same folder.")
        st.stop()

data = load_data()

# Prepare data
X = data.drop(columns=["Adherence"])
y = data["Adherence"].apply(lambda x: 1 if x == "Adherent" else 0)

# Train model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(X, y)

# Input fields
st.subheader("Enter Patient Details")
gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
age = st.number_input("Age", min_value=0, max_value=120)
medication_type = st.selectbox("Medication Type", ["Type1", "Type2"])
missed_doses = st.number_input("Missed Doses", min_value=0)
last_visit_gap = st.number_input("Days since Last Visit", min_value=0)
app_usage = st.selectbox("App Usage", ["Yes", "No"])

# Encode inputs
gender_encoded = 0 if gender == "Male" else (1 if gender == "Female" else 2)
medication_encoded = 0 if medication_type == "Type1" else 1
app_usage_encoded = 1 if app_usage == "Yes" else 0

# Make prediction
features = [[gender_encoded, age, medication_encoded, missed_doses, last_visit_gap, app_usage_encoded]]

if st.button("Predict"):
    prediction = model.predict(features)[0]
    st.success("Adherent ✅" if prediction == 1 else "Non-Adherent ❌")
