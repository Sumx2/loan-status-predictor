# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("loan_status_model.pkl")

st.title("🏦 Loan Status Prediction App")

# Input form
with st.form("loan_form"):
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount", min_value=0)
    Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # Encode inputs
    Gender = 1 if Gender == "Male" else 0
    Married = 1 if Married == "Yes" else 0
    Dependents = 4 if Dependents == "3+" else int(Dependents)
    Education = 1 if Education == "Graduate" else 0
    Self_Employed = 1 if Self_Employed == "Yes" else 0
    Property_Area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[Property_Area]

    input_data = pd.DataFrame([[
        Gender, Married, Dependents, Education, Self_Employed,
        ApplicantIncome, CoapplicantIncome, LoanAmount,
        Loan_Amount_Term, Credit_History, Property_Area
    ]], columns=[
        "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Loan_Amount_Term", "Credit_History", "Property_Area"
    ])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
