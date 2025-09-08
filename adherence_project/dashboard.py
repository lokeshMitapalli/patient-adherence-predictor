import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from twilio.rest import Client

st.title("Medication Adherence Predictor (Upload Dataset + SMS Alerts)")

# ✅ File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    if "Adherence" not in data.columns:
        st.error("Dataset must contain an 'Adherence' column.")
        st.stop()

    # ✅ Preprocessing
    X = data.drop(columns=["Adherence"])
    y = data["Adherence"].apply(lambda x: 1 if x == "Adherent" else 0)
    X = pd.get_dummies(X, drop_first=True)

    # ✅ Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    st.success("Model trained successfully!")

    # ✅ Patient details for prediction
    st.subheader("Enter Patient Details")
    gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
    age = st.number_input("Age", min_value=0, max_value=120)
    medication_type = st.selectbox("Medication Type", ["Type1", "Type2"])
    missed_doses = st.number_input("Missed Doses", min_value=0)
    last_visit_gap = st.number_input("Days since Last Visit", min_value=0)
    app_usage = st.selectbox("App Usage", ["Yes", "No"])

    input_data = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Medication_Type": [medication_type],
        "Missed_Doses": [missed_doses],
        "Days_Since_Last_Visit": [last_visit_gap],
        "App_Usage": [app_usage]
    })

    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.success("Prediction: Adherent ✅")
        else:
            st.error("Prediction: Non-Adherent ❌")

            # ✅ SMS Notification
            phone_number = st.text_input("Enter patient's phone number (e.g., +91XXXXXXXXXX)")
            if st.button("Send SMS Reminder"):
                if phone_number:
                    try:
                        account_sid = "YOUR_TWILIO_SID"
                        auth_token = "YOUR_TWILIO_AUTH_TOKEN"
                        client = Client(account_sid, auth_token)

                        message = client.messages.create(
                            body="Reminder: Please take your medication on time!",
                            from_="+1XXXXXXXXXX",  # Your Twilio phone number
                            to=phone_number
                        )
                        st.success(f"SMS sent successfully! SID: {message.sid}")
                    except Exception as e:
                        st.error(f"Failed to send SMS: {e}")
                else:
                    st.warning("Please enter a valid phone number.")
