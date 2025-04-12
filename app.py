# # app.py
# import streamlit as st
# import pandas as pd
# import joblib

# # Load model
# model = joblib.load("loan_status_model.pkl")

# st.title("🏦 Loan Status Prediction App")

# # Input form
# with st.form("loan_form"):
#     Gender = st.selectbox("Gender", ["Male", "Female"])
#     Married = st.selectbox("Married", ["Yes", "No"])
#     Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
#     Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
#     Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
#     ApplicantIncome = st.number_input("Applicant Income", min_value=0)
#     CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
#     LoanAmount = st.number_input("Loan Amount", min_value=0)
#     Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
#     Credit_History = st.selectbox("Credit History", [1.0, 0.0])
#     Property_Area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
    
#     submitted = st.form_submit_button("Predict")

# if submitted:
#     # Encode inputs
#     Gender = 1 if Gender == "Male" else 0
#     Married = 1 if Married == "Yes" else 0
#     Dependents = 4 if Dependents == "3+" else int(Dependents)
#     Education = 1 if Education == "Graduate" else 0
#     Self_Employed = 1 if Self_Employed == "Yes" else 0
#     Property_Area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[Property_Area]

#     input_data = pd.DataFrame([[
#         Gender, Married, Dependents, Education, Self_Employed,
#         ApplicantIncome, CoapplicantIncome, LoanAmount,
#         Loan_Amount_Term, Credit_History, Property_Area
#     ]], columns=[
#         "Gender", "Married", "Dependents", "Education", "Self_Employed",
#         "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
#         "Loan_Amount_Term", "Credit_History", "Property_Area"
#     ])

#     prediction = model.predict(input_data)

#     if prediction[0] == 1:
#         st.success("✅ Loan Approved")
#     else:
#         st.error("❌ Loan Not Approved")


# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Loan Status Predictor", layout="centered")

# Load model and data
model = joblib.load("loan_status_model.pkl")
df = pd.read_csv("loan_dataset.csv").dropna()

# Preprocessing
df.replace({"Loan_Status": {"N": 0, "Y": 1},
            "Married": {"Yes": 1, "No": 0},
            "Gender": {"Male": 1, "Female": 0},
            "Self_Employed": {"No": 0, "Yes": 1},
            "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
            "Education": {"Graduate": 1, "Not Graduate": 0}}, inplace=True)
df["Dependents"] = df["Dependents"].replace("3+", 4).astype(int)

# Title
st.title("🏦 Loan Status Prediction App")

# Tabs: Prediction | Data Visualizations
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Data Visualizations"])

# =================== Tab 1: Prediction =================== #
with tab1:
    st.subheader("Enter Loan Details")
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

        input_data = pd.DataFrame([[Gender, Married, Dependents, Education, Self_Employed,
                                    ApplicantIncome, CoapplicantIncome, LoanAmount,
                                    Loan_Amount_Term, Credit_History, Property_Area]],
                                  columns=["Gender", "Married", "Dependents", "Education", "Self_Employed",
                                           "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                                           "Loan_Amount_Term", "Credit_History", "Property_Area"])
        
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Not Approved")

# =================== Tab 2: Visualizations =================== #
with tab2:
    st.subheader("Explore the Loan Dataset")

    # Countplot: Education vs Loan Status
    st.write("### 🎓 Education vs Loan Approval")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Education", hue="Loan_Status", data=df, ax=ax1)
    ax1.set_xticklabels(["Not Graduate", "Graduate"])
    st.pyplot(fig1)

    # Countplot: Married vs Loan Status
    st.write("### 💍 Marital Status vs Loan Approval")
    fig2, ax2 = plt.subplots()
    sns.countplot(x="Married", hue="Loan_Status", data=df, ax=ax2)
    ax2.set_xticklabels(["Not Married", "Married"])
    st.pyplot(fig2)

    # Correlation Heatmap
    # st.write("### 🔥 Feature Correlation")
    # fig3, ax3 = plt.subplots(figsize=(10, 6))
    # sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax3)
    # st.pyplot(fig3)


    # Distribution of Loan Amount
    st.write("### 💰 Loan Amount Distribution")
    fig4, ax4 = plt.subplots()
    sns.histplot(df["LoanAmount"], kde=True, color="green", ax=ax4)
    st.pyplot(fig4)

    # Pie Chart of Loan Status
    st.write("### 📊 Loan Status Distribution")
    loan_status_counts = df["Loan_Status"].value_counts()
    fig5, ax5 = plt.subplots()
    ax5.pie(loan_status_counts, labels=["Approved", "Not Approved"], autopct="%1.1f%%", startangle=90, colors=["#00c49a", "#ff6b6b"])
    ax5.axis("equal")
    st.pyplot(fig5)
