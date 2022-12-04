import pickle
import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,  OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
#from imblearn.over_sampling import SMOTE
import numpy as np
# with open('model','rb') as f:
#     model = pickle.load(f)
sam = joblib.load('pipe1.joblib')
feature_names=['Gender', 'Married', 'Dependents', 'Education','Self_Employed','ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
        # append all inputs into a single list
        #X_arr = np.array([X])  # convert list to numpy array

#Loan_ID = 'LP001116'     
Gender = 'Male'   
Married = 'Yes'  
Dependents = '1' 
Education = 'Graduate'     
Self_Employed = 'No'   
Property_Area = 'Rural'  
ApplicantIncome = 4538 
CoapplicantIncome = 1508      
LoanAmount = 128.0  
Loan_Amount_Term = 360.0 
Credit_History = 1.0
values=[Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]        
x=pd.DataFrame(data=[values],columns=feature_names)

x.ApplicantIncome = np.sqrt(x.ApplicantIncome)
x.CoapplicantIncome = np.sqrt(x.CoapplicantIncome)
x.LoanAmount = np.sqrt(x.LoanAmount)

col = make_column_transformer(
    (OneHotEncoder(), ['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']),
    remainder ='passthrough')
#LRclassifier = LogisticRegression(solver='saga', max_iter=500, random_state=1)

# pipe = make_pipeline(column_trans,model)
new_vals= pd.DataFrame([['Female','No','0','Not Graduate','No',0,50,1100.0,360.0,1.0,'Semiurban']],columns = feature_names)

pre=sam.predict(x)
print(pre)
# prediction = model.predict(x)
# output = prediction[0]
# # prediction = model.predict(output)
# print(prediction)