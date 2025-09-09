import streamlit as st
import pandas as pd
import pickle
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

st.title("Patient Adherence Prediction Dashboard")

# Option to upload dataset
st.sidebar.header("Upload Your Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset Preview")
    st.dataframe(data.head())
else:
    default_path = os.path.join(os.path.dirname(__file__), 'patient_adherence_dataset.csv')
    data = pd.read_csv(default_path)
    st.write("### Default Dataset Preview")
    st.dataframe(data.head())

# Handle Adherence column safely
if "Adherence" in data.columns:
    y = data["Adherence"].fillna("").apply(lambda x: 1 if str(x).strip().lower() == "adherent" else 0)
else:
    st.error("The dataset must contain an 'Adherence' column.")
    st.stop()

# Get feature columns (exclude Adherence)
X = data.drop(columns=["Adherence"], errors='ignore')

# Prediction section
st.sidebar.header("Make a Prediction")
input_data = {}
for col in X.columns:
    value = st.sidebar.text_input(f"Enter {col}")
    input_data[col] = value

if st.sidebar.button("Predict"):
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Match column types with training data
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except:
                pass

        prediction = model.predict(input_df)[0]
        result = "Adherent" if prediction == 1 else "Non-Adherent"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

