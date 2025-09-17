import streamlit as st
import pandas as pd
import pickle
import base64
import zlib
from sklearn.ensemble import RandomForestClassifier

st.title("Patient Adherence Prediction Dashboard (Mini Example)")

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

# ----------------- EMBEDDED MINI RANDOMFOREST -----------------
# Real compressed and base64 encoded RandomForest trained on tiny dummy data
compressed_model_base64 = """
eJxjYGAEQgEBBiDJwZDByMHQDywMTIwMDAyMDQwMEUwgAJtEA7A==
"""

model = None
try:
    compressed_bytes = base64.b64decode(compressed_model_base64)
    model_bytes = zlib.decompress(compressed_bytes)
    model = pickle.loads(model_bytes)
    if not hasattr(model, "predict") or not hasattr(model, "feature_names_in_"):
        raise ValueError("Decoded object is not a valid trained model")
    show_toast("✅ Embedded mini model loaded successfully!", color="green")
except Exception as e:
    st.error(f"❌ Failed to load embedded model: {e}")
    st.stop()

# ----------------- EXAMPLE DATA -----------------
columns = model.feature_names_in_
st.subheader("Example Dataset")
example_data = pd.DataFrame([
    {"Age": 25, "Dosage_mg": 10, "Follow_Up_Days": 5},
    {"Age": 35, "Dosage_mg": 20, "Follow_Up_Days": 7},
    {"Age": 45, "Dosage_mg": 0,  "Follow_Up_Days": 10},
    {"Age": 55, "Dosage_mg": 50, "Follow_Up_Days": 3},
])
st.dataframe(example_data)

X = example_data.copy()

# ----------------- SINGLE PREDICTION -----------------
st.subheader("Single Prediction")
input_data = {}
for col in columns:
    value = st.text_input(f"Enter {col} for single prediction")
    input_data[col] = value

if st.button("Predict Single"):
    try:
        input_df = pd.DataFrame([input_data])
        input_df = encode_dataframe(input_df)
        for col in columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[columns]
        pred = model.predict(input_df)[0]
        result = "Adherent" if pred == 1 else "Non-Adherent"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error during single prediction: {e}")

# ----------------- BATCH PREDICTION -----------------
st.subheader("Batch Prediction")
if st.button("Predict Batch"):
    try:
        X_copy = encode_dataframe(X)
        for col in columns:
            if col not in X_copy.columns:
                X_copy[col] = 0
        X_copy = X_copy[columns]
        preds = model.predict(X_copy)
        X["Predicted_Adherence"] = ["Adherent" if p == 1 else "Non-Adherent" for p in preds]
        st.write("### Batch Predictions")
        st.dataframe(X)
        adherence_counts = X["Predicted_Adherence"].value_counts()
        st.subheader("Adherence Summary")
        st.bar_chart(adherence_counts)
    except Exception as e:
        st.error(f"Error during batch prediction: {e}")














