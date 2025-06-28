
import warnings
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀")
st.title("🫀 Heart Disease Prediction App")

# Load the trained pipeline (preprocessing + model)
try:
    model = joblib.load('E:/Ali Elsayed/Sprints/Heart_Disease_Project/models/final_model.pkl')
except FileNotFoundError:
    st.error("❌ Model file not found. Make sure '../models/final_model.pkl' exists.")
    st.stop()

st.write("### Enter patient data below:")

# Raw input fields (these must match the original dataset format)
user_input = {
    'age': st.number_input('Age', min_value=1, max_value=120, value=50),
    'sex': st.selectbox('Sex (1 = Male, 0 = Female)', [1, 0]),
    'cp': st.selectbox('Chest Pain Type (0–3)', [0, 1, 2, 3]),
    'trestbps': st.number_input('Resting Blood Pressure (mm Hg)', value=120),
    'chol': st.number_input('Cholesterol (mg/dl)', value=200),
    'fbs': st.selectbox('Fasting Blood Sugar > 120 (1 = True, 0 = False)', [0, 1]),
    'restecg': st.selectbox('Resting ECG Results (0–2)', [0, 1, 2]),
    'thalach': st.number_input('Max Heart Rate Achieved', value=150),
    'exang': st.selectbox('Exercise Induced Angina (1 = Yes, 0 = No)', [0, 1]),
    'oldpeak': st.number_input('ST Depression', value=1.0, format="%.1f"),
    'slope': st.selectbox('Slope of ST Segment (0–2)', [0, 1, 2]),
    'ca': st.selectbox('Number of Major Vessels (0–3)', [0, 1, 2, 3]),
    'thal': st.selectbox('Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)', [1, 2, 3])
}

if st.button("🔍 Predict"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Use the full pipeline to predict
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]  # Probability of class 1

    if prediction == 1:
        st.error(f"💔 Likely Heart Disease Detected! (Risk: {prediction_proba:.2%})")
    else:
        st.success(f"❤️ No Heart Disease Detected (Risk: {prediction_proba:.2%})")
