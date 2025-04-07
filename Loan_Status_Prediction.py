


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib






loan_dataset=pd.read_csv('loan_dataset.csv')



loan_dataset



type(loan_dataset)



loan_dataset.head()



loan_dataset.tail()




loan_dataset.info()



loan_dataset.shape


# Statistical Information of all Numerical features



loan_dataset.describe()


# No of missing values


loan_dataset.isnull().sum()


# Dropping the missing values
# 



loan_dataset=loan_dataset.dropna()



loan_dataset.isnull().sum()



loan_dataset.shape


# Label Encoding


loan_dataset.replace({"Loan_Status":{"N":0,"Y":1}},inplace=True)


loan_dataset.head()


loan_dataset["Dependents"].value_counts()


loan_dataset=loan_dataset.replace(to_replace='3+',value=4)

loan_dataset["Dependents"].value_counts()


# Data Visualisation

#Education and loan status

sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)


#fro marital status and loan status

sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset)

loan_dataset.replace({"Married": {"Yes": 1, "No": 0},
                      "Gender": {"Male": 1, "Female": 0},
                      "Self_Employed": {"No": 0, "Yes": 1},
                      "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
                      "Education": {"Graduate": 1, "Not Graduate": 0}},
                     inplace=True)



loan_dataset.head()


# Seaparating X and Y

X= loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y= loan_dataset['Loan_Status']


print(X)



print(Y)


# Train test Split


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y,random_state=2)




print(X.shape, X_train.shape, X_test.shape)


# training model: SVM



classifier = svm.SVC(kernel="linear")



classifier.fit(X_train, Y_train)


# model evaluation



# Save the model
joblib.dump(classifier, 'loan_status_model.pkl')

print("Model saved successfully!")


import joblib
joblib.dump(classifier, 'loan_status_model.pkl')



#accuracy score on training data
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print("Accuracy on training data: ", training_data_accuracy)


#accuracy on test data
X_test_prediction=classifier.predict(X_test)
test_data_acccuracy=accuracy_score(X_test_prediction,Y_test)
print("Accuracy on test data: ", test_data_acccuracy)


# Making a predictive system



def predict_loan_status():
  Gender = int(input("Enter Gender (1 for Male, 0 for Female): "))
  Married = int(input("Enter Marital Status (1 for Married, 0 for Not Married): "))
  Dependents = int(input("Enter Number of Dependents (0, 1, 2, or 4): "))
  Education = int(input("Enter Education (1 for Graduate, 0 for Not Graduate): "))
  Self_Employed = int(input("Enter Self-Employed Status (1 for Yes, 0 for No): "))
  ApplicantIncome = float(input("Enter Applicant Income: "))
  CoapplicantIncome = float(input("Enter Coapplicant Income: "))
  LoanAmount = float(input("Enter Loan Amount: "))
  Loan_Amount_Term = float(input("Enter Loan Amount Term: "))
  Credit_History = float(input("Enter Credit History (1 for Good, 0 for Bad): "))
  Property_Area = int(input("Enter Property Area (0 for Rural, 1 for Semiurban, 2 for Urban): "))

  input_data = pd.DataFrame({
      'Gender': [Gender],
      'Married': [Married],
      'Dependents': [Dependents],
      'Education': [Education],
      'Self_Employed': [Self_Employed],
      'ApplicantIncome': [ApplicantIncome],
      'CoapplicantIncome': [CoapplicantIncome],
      'LoanAmount': [LoanAmount],
      'Loan_Amount_Term': [Loan_Amount_Term],
      'Credit_History': [Credit_History],
      'Property_Area': [Property_Area]
  })
  prediction = classifier.predict(input_data)
  if prediction[0] == 1:
    print("Loan Approved")
  else:
    print("Loan Not Approved")

predict_loan_status()