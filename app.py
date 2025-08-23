import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load encoders
with open("encoders.pkl", "rb") as file:
    encoders = pickle.load(file)

# Load trained model
with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

    loaded_model = loaded_model['model']

st.title("Customer Churn Prediction App")

# Collect user inputs
age = st.number_input("Age", min_value=18, max_value=100)
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=120)
usage_freq = st.number_input("Usage Frequency", min_value=0, max_value=100)
support_calls = st.number_input("Support Calls", min_value=0, max_value=50)
payment_delay = st.number_input("Payment Delay (days)", min_value=0, max_value=365)
subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
total_spend = st.number_input("Total Spend", min_value=0, max_value=100000)
last_interaction = st.number_input("Last Interaction (days ago)", min_value=0, max_value=365)

# Convert input into DataFrame
input_data = {
    "Age": age,
    "Gender": gender,
    "Tenure": tenure,
    "Usage Frequency": usage_freq,
    "Support Calls": support_calls,
    "Payment Delay": payment_delay,
    "Subscription Type": subscription_type,
    "Contract Length": contract_length,
    "Total Spend": total_spend,
    "Last Interaction": last_interaction
}

input_data_df = pd.DataFrame([input_data])

# Encode categorical variables
for col, encoder in encoders.items():
    if col in input_data_df.columns:
        input_data_df[col] = encoder.transform(input_data_df[col])

# Prediction
if st.button("Predict"):
    prediction = loaded_model.predict(input_data_df)
    pred_prob = loaded_model.predict_proba(input_data_df)

    st.subheader("Prediction Result")
    st.write(f"Predicted Class: **{prediction[0]}**")

    st.subheader("Prediction Probabilities")
    for i, prob in enumerate(pred_prob.flatten()):
        st.write(f"Class {i}: {(prob)*100:.1f}%")
