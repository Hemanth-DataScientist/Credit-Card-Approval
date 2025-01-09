import streamlit as st
import joblib
import numpy as np

# Load the saved ML model
model = joblib.load('model.pkl')

# Streamlit UI
st.title("📊 Credit Card Approval Prediction")
st.write("Enter details below to check if a credit card application will be approved or rejected.")

# Input fields
amt_income = st.number_input("Total Income (in USD)", min_value=0)
cnt_children = st.number_input("Number of Children", min_value=0, step=1)
flag_own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
flag_own_realty = st.selectbox("Owns Realty?", ["Yes", "No"])
days_birth = st.number_input("Age in Days (Negative Value)", min_value=-36500, max_value=-6570)
days_employed = st.number_input("Days Employed (Negative or Positive)", min_value=-36500, max_value=36500)

# Convert categorical inputs to numerical
flag_own_car = 1 if flag_own_car == "Yes" else 0
flag_own_realty = 1 if flag_own_realty == "Yes" else 0

# Predict Button
if st.button("Predict Approval"):
    input_features = np.array([[amt_income, cnt_children, flag_own_car, flag_own_realty, days_birth, days_employed]])
    prediction = model.predict(input_features)

    result = "✅ Approved" if prediction[0] == 1 else "❌ Rejected"
    st.success(f"Prediction: {result}")
