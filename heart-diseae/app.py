import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
with open(r"model.pkl", "rb") as model_file:
    model = joblib.load(model_file)

# Title
st.title("Heart Disease Prediction App")
st.markdown("Enter patient data to predict presence of heart disease.")

# Input form
with st.form("heart_form"):
    age = st.number_input("Age", 1, 120, step=1)
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, step=1)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, step=1)
    thalch = st.number_input("Max Heart Rate Achieved", 60, 220, step=1)
    oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, step=0.1)

    sex = st.selectbox("Sex", ["Male", "Female"])
    dataset = st.selectbox("Dataset Origin", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])
    cp = st.selectbox("Chest Pain Type", ["asymptomatic", "typical angina", "atypical angina", "non-anginal"])
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True", "nan"])
    restecg = st.selectbox("Resting ECG", ["normal", "left ventricular hypertrophy", "st-t abnormality"])
    exang = st.selectbox("Exercise Induced Angina", ["False", "True"])
    slope = st.selectbox("Slope of ST Segment", ["upsloping", "flat", "downsloping", "nan"])
    ca = st.selectbox("Number of Major Vessels (ca)", ["0.0", "1.0", "2.0", "3.0", "nan"])
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversible defect", "nan"])

    submitted = st.form_submit_button("Predict")

# Build input features only if form is submitted
if submitted:
    input_dict = {
        'age': age,
        'trestbps': trestbps,
        'chol': chol,
        'thalch': thalch,
        'oldpeak': oldpeak,
        'num': 0,  # Dummy placeholder if needed
        'sex_Male': 1 if sex == "Male" else 0,
        'dataset_Hungary': 1 if dataset == "Hungary" else 0,
        'dataset_Switzerland': 1 if dataset == "Switzerland" else 0,
        'dataset_VA Long Beach': 1 if dataset == "VA Long Beach" else 0,
        'cp_atypical angina': 1 if cp == "atypical angina" else 0,
        'cp_non-anginal': 1 if cp == "non-anginal" else 0,
        'cp_typical angina': 1 if cp == "typical angina" else 0,
        'fbs_True': 1 if fbs == "True" else 0,
        'fbs_nan': 1 if fbs == "nan" else 0,
        'restecg_normal': 1 if restecg == "normal" else 0,
        'restecg_st-t abnormality': 1 if restecg == "st-t abnormality" else 0,
        'exang_True': 1 if exang == "True" else 0,
        'slope_flat': 1 if slope == "flat" else 0,
        'slope_nan': 1 if slope == "nan" else 0,
        'slope_upsloping': 1 if slope == "upsloping" else 0,
        'ca_1.0': 1 if ca == "1.0" else 0,
        'ca_2.0': 1 if ca == "2.0" else 0,
        'ca_3.0': 1 if ca == "3.0" else 0,
        'thal_nan': 1 if thal == "nan" else 0,
        'thal_normal': 1 if thal == "normal" else 0,
    }

    # Ensure all 26 features are included
    feature_order = [
        'age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'num',
        'sex_Male', 'dataset_Hungary', 'dataset_Switzerland', 'dataset_VA Long Beach',
        'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
        'fbs_True', 'fbs_nan', 'restecg_normal', 'restecg_st-t abnormality',
        'exang_True', 'slope_flat', 'slope_nan', 'slope_upsloping',
        'ca_1.0', 'ca_2.0', 'ca_3.0', 'thal_nan', 'thal_normal'
    ]

    # Fill missing values with 0 if any
    for feature in feature_order:
        input_dict.setdefault(feature, 0)

    input_df = pd.DataFrame([input_dict], columns=feature_order)

    # Make prediction
    prediction = model.predict(input_df)[0]
    result = "💖 No Heart Disease Detected" if prediction == 0 else "⚠️ Risk of Heart Disease Detected"

    st.subheader("Prediction Result")
    st.success(result)
