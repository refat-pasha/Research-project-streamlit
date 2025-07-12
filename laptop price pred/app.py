import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and categorical choices
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_laptop_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    choices = joblib.load('categorical_choices.pkl')
    return model, scaler, choices

model, scaler, choices = load_artifacts()

st.title("Laptop Price Prediction")
st.write("Enter laptop specifications to predict the price (Euros).")

# Sidebar for user input
st.sidebar.header("Laptop Specifications")

company = st.sidebar.selectbox("Company", choices['Company'])
typename = st.sidebar.selectbox("TypeName", choices['TypeName'])
inches = st.sidebar.number_input("Screen Size (inches)", min_value=10.0, max_value=20.0, value=15.6)
ram = st.sidebar.number_input("RAM (GB)", min_value=2, max_value=128, value=8)
os = st.sidebar.selectbox("Operating System", choices['OS'])
weight = st.sidebar.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
screenres = st.sidebar.selectbox("Screen Resolution", choices['ScreenResolution'])
cpu = st.sidebar.selectbox("CPU", choices['Cpu'])
memory = st.sidebar.selectbox("Memory", choices['Memory'])
gpu = st.sidebar.selectbox("GPU", choices['Gpu'])

# Prepare input DataFrame
input_dict = {
    'Company': [company],
    'TypeName': [typename],
    'Inches': [inches],
    'Ram': [ram],
    'OS': [os],
    'Weight': [weight],
    'ScreenResolution': [screenres],
    'Cpu': [cpu],
    'Memory': [memory],
    'Gpu': [gpu]
}
input_df = pd.DataFrame(input_dict)

# Encode categorical features to match training
for col in choices:
    input_df[col] = input_df[col].astype(str)
    # Map to integer codes as in training
    input_df[col] = input_df[col].apply(lambda x: choices[col].index(x) if x in choices[col] else 0)

# Scale numeric features
X_scaled = scaler.transform(input_df)

# Prediction
if st.sidebar.button("Predict Price"):
    pred = model.predict(X_scaled)[0]
    st.subheader(f"Predicted Price: €{pred:,.2f}")

# Optionally show input data
if st.checkbox("Show input data"):
    st.write(input_df)
