# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import joblib

# Load and preprocess data
df = pd.read_csv("loan_dataset.csv")
df = df.dropna()
df.replace({"Loan_Status": {"N": 0, "Y": 1},
            "Dependents": {"3+": 4},
            "Married": {"Yes": 1, "No": 0},
            "Gender": {"Male": 1, "Female": 0},
            "Self_Employed": {"No": 0, "Yes": 1},
            "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
            "Education": {"Graduate": 1, "Not Graduate": 0}}, inplace=True)

X = df.drop(columns=["Loan_ID", "Loan_Status"])
y = df["Loan_Status"]

# Train model
model = svm.SVC(kernel="linear")
model.fit(X, y)

# Save model
joblib.dump(model, "loan_status_model.pkl")
