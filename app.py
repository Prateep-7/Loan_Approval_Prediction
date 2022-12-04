from flask import Flask, request, render_template
from flask_cors import cross_origin
from flask import Response
import pickle
import pandas as pd

app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates')

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/applyLoan", methods=["GET", "POST"])
@cross_origin()
def applyLoan():
    if request.method == "POST":
        loan_amount_form = int(request.form['loan_amount'])
        loan_amount_term_form = int(request.form['loan_amount_term'])
        applicant_income_form = int(request.form['applicant_income'])
        coapplicant_income_form = int(request.form['coapplicant_income'])
        
        credit_history_form = int(request.form['credit_history'])
        #if credit_history_form
        education_form = request.form['education']
        if education_form == 'Graduate':
            education_form=1
        else:
            education_form=0

        self_employed_form = request.form['self_employed']
        if self_employed_form=='Yes':
            self_employed_form=1
        else:
            self_employed_form=0

        gender_form = request.form['gender']
        if gender_form=='Male':
            gender_form=1
        else:
            gender_form=0

        married_form = request.form['married']
        if married_form=='Yes':
            married_form=1
        else:
            married_form=0
        
        #Property_Area_Rural,Property_Area_Urban,Property_Area_Semiurban
        property_area = request.form['property_area']
        if property_area=='Rural':
            Property_Area_Rural=1
            Property_Area_Semiurban=0
            Property_Area_Urban=0
        elif property_area=='Urban':
            Property_Area_Rural=0
            Property_Area_Semiurban=0
            Property_Area_Urban=1
        else:
            Property_Area_Rural=0
            Property_Area_Semiurban=1
            Property_Area_Urban=0

        #Dependents_0,Dependents_1,Dependents_2,Dependents_3    
        dependents_form = request.form['dependents']
        if dependents_form=='0':
            Dependents_0=1
            Dependents_1=0
            Dependents_2=0
            Dependents_3=0
        elif dependents_form=='1':
            Dependents_0=0
            Dependents_1=1
            Dependents_2=0
            Dependents_3=0
        elif dependents_form=='2':
            Dependents_0=0
            Dependents_1=0
            Dependents_2=1
            Dependents_3=0
        else:
            Dependents_0=0
            Dependents_1=0
            Dependents_2=0
            Dependents_3=1


        loan_amount_list = [loan_amount_form]
        loan_amount_term_list = [loan_amount_term_form]
        applicant_income_list = [applicant_income_form]
        coapplicant_income_list = [coapplicant_income_form]
        credit_history_list = [credit_history_form]

        education_list = [education_form]
        self_employed_list = [self_employed_form]
        gender_list = [gender_form]
        married_list = [married_form]
        
        property_arear = [Property_Area_Rural]
        property_areau =[Property_Area_Urban] 
        property_areas =[Property_Area_Semiurban]
        dependents0 = [Dependents_0]
        dependents1 = [Dependents_1]
        dependents2 = [Dependents_2]
        dependents3 = [Dependents_3]
        #16 attributes
        feature_names=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Gender', 'Married',
       'Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3',
       'Education', 'Self_Employed', 'Property_Area_Rural',
       'Property_Area_Semiurban', 'Property_Area_Urban']
        values =  applicant_income_list + coapplicant_income_list + loan_amount_list + loan_amount_term_list + credit_history_list + gender_list + married_list + dependents0 + dependents1 + dependents2 + dependents3 + education_list + self_employed_list + property_arear + property_areas + property_areau
        X=pd.DataFrame(data=[values],columns=feature_names)
        
        with open('A:/Programs/Mini Project/Final/UI-2/Model/model4','rb') as f:
            model = pickle.load(f)
        y_pred=model.predict(X)
        
        if y_pred == 1:
            final_pred =  'LOAN APPROVED!'
        else:
            final_pred = 'SORRY, YOUR LOAN HAS BEEN DENIED! '
    
        return render_template('apply.html', prediction_text=final_pred)

    return render_template('apply.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)  