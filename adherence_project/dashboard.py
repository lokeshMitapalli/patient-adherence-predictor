import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from twilio.rest import Client

st.title("Medication Adherence Predictor with SMS Alerts")

# ✅ Load dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("adherence_project/patient_adherence_dataset.csv")
        return data
    except FileNotFoundError:
        st.error("Dataset not found! Please upload patient_adherence_dataset.csv in the same folder.")
        st.stop()

data = load_data()

# ✅ Preprocess dataset
X = data.drop(columns=["Adherence"])
y = data["Adherence"].apply(lambda x: 1 if x == "Adherent" else 0)

X = pd.get_dummies(X, drop_first=True)

# ✅ Train model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns

model, feature_cols = train_model(X, y)

# ✅ User Input Form
st.subheader("Enter Patient Details")
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

# ✅ Predict and Show Result
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("Prediction: Adherent ✅")
    else:
        st.error("Prediction: Non-Adherent ❌")

        # ✅ SMS Notification Section
        st.subheader("Send SMS Notification")
        phone_number = st.text_input("Enter patient's phone number (e.g., +91XXXXXXXXXX)")
        if st.button("Send SMS Reminder"):
            if phone_number:
                # ✅ Twilio Credentials
                account_sid = "YOUR_TWILIO_ACCOUNT_SID"
                auth_token = "YOUR_TWILIO_AUTH_TOKEN"
                client = Client(account_sid, auth_token)

                message = client.messages.create(
                    body="Reminder: Please take your medication on time!",
                    from_="+1XXXXXXXXXX",  # Your Twilio number
                    to=phone_number
                )
                st.success(f"SMS sent successfully! SID: {message.sid}")
            else:
                st.warning("Please enter a phone number.")
