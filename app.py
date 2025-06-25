from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np
from flask_cors import CORS
# Update your functions to work with individual values:
def convert_emp_str(x):
    if x<1:
        return '< 1 year'
    elif x==1:
        return '1 year'
    elif x<10:
        return str(int(x)) + ' years'
    else :
        return '10+ years'

def extract_quarter_single(date_str):
    """Extract quarter from a single date string"""
    try:
        period = pd.to_datetime(date_str, format='%b-%Y').to_period('Q')
        return period.ordinal
    except:
        return None

def extract_month_single(date_str):
    """Extract month from a single date string"""
    try:
        return pd.to_datetime(date_str, format='%b-%Y').month
    except:
        return None
def emp_len_bin(x):
    if x < 1.5:
        return '1 yr'
    if 1.5 <= x < 7.5 :
        return '2-7 yrs'
    if 7.5 <= x < 9.5:
        return '8-9 yrs'
    if x >= 9.5 :
        return '10+ yrs'
def empl_length_num_single(emp_length_str):
    """Convert employment length string to numeric for a single value"""
    # You'll need to adapt your original empl_length_num function here
    # This is just an example - replace with your actual logic
    if pd.isna(emp_length_str):
        return None
    elif '10+' in str(emp_length_str):
        return 10
    elif '<' in str(emp_length_str):
        return 0
    else:
        return int(str(emp_length_str).split()[0])
def purpose_bin(val):
    if val in ['credit_card','home_improvement','major_purchase','educational','wedding','car','vacation','house']:
        return 'family'
    else:
        if val != 'small_business' :
            return 'other'
        else :
            return 'small_business'
           
def purpose_apply(x):
    return x.iloc[:, 0].apply(purpose_bin).to_frame()

def home_own_bin(val):
    if val not in ['MORTGAGE','OWN','RENT']:
        return 'RENT' 
    else:
        return val
def zip_extract(x):
    return x.iloc[:, 0].str.split().str[-1].to_frame()
# Updated DataFrame wrapper functions:
def extract_quarter_df(x):
    return x.iloc[:, 0].apply(extract_quarter_single).to_frame()

def extract_month_df(x):
    return x.iloc[:, 0].apply(extract_month_single).to_frame()

def empl_length_num_df(x):
    return x.iloc[:, 0].apply(empl_length_num_single).to_frame()

def home_ownership_cleanup_df(x):
    return x.iloc[:, 0].apply(home_own_bin).to_frame()

def emp_bin_ext(x):
    return pd.DataFrame(x).iloc[:, 0].apply(emp_len_bin).to_frame()
def clip_trans(x,upper = None,lower = None):
    return x.iloc[:, 0].clip(upper,lower).to_frame()
def clip_trans_df(x,upper=None,lower = None):
    return pd.DataFrame(x).iloc[:, 0].clip(upper,lower).to_frame()


app = Flask(__name__)
CORS(app)

# Load your trained model and preprocessor
model = joblib.load('notebooks/pipeline_1.joblib')
# scaler = joblib.load('models/scaler.pkl')  # if you used scaling

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        print(data)
        # Extract features (adjust based on your model features)
        features = {
            'annual_inc': float(data['annual_income']),
            'loan_amnt': float(data['loan_amount']),
            'dti': float(data['debt_to_income']),
            'term' : ' ' + data['loan_term'] + ' months',
            'int_rate' : float(data['interest_rate']),
            'home_ownership' : data['home_ownership'],
            'purpose' : data['purpose'],
            'mort_acc' : float(data['mortgage_accounts']),
            'grade' : 'A',
            'sub_grade' : 'A1',
            'title' : 'Debt',
            'emp_length' : convert_emp_str(float(data['emp_length'])),
            'verification_status' : data['verification_status'],
            'issue_d' : "Oct-2014",
            'earliest_cr_line' : (pd.to_datetime("Oct-2014",format = "%b-%Y")+pd.DateOffset(years = int(data['emp_length']))).strftime("%b-%Y"),
            'open_acc' : float(data['open_acc']),
            'pub_rec_bankruptices' : float(data['pub_rec_bankruptcies']),
            'revol_util' : float(data['revol_util']) , 
            'revol_bal' : float(data['revol_bal']),
            'intial_list_status' : 'w',
            'application_status' : 'INDIVIDUAL' , 
            'address' : np.random.choice(['11650','22690','30723','48052','70466'])
            # Add other features your model uses
        }
        
        # Create DataFrame
        features['installment'] = features['loan_amnt']*(features['int_rate']/1200.0)*(1+features['int_rate']/1200.0)**float(features['term'][:2])/((1+features['int_rate']/1200.0)**float(features['term'][:2])-1)
        features['total_acc'] = round(features['open_acc']*10/7,0)
        features['pub_rec'] = features['pub_rec_bankruptices']
        input_df = pd.DataFrame([features])
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(input_df)
        # Make prediction
        probability = model.predict_proba(input_df)[0][1]  # Probability of default
        prediction = model.predict(input_df)[0]  # 0 or 1
        # Get feature importance (if using logistic regression)
        feature_importance = dict(zip(
            model[-2].get_fea,
            abs(model[-1].coef_)
        ))
        
        # Sort by importance
        top_factors = sorted(feature_importance.items(), 
                           key=lambda x: x[1], 
                           reverse=True)[:3]
        
        response = {
            'default_probability': round(float(probability), 4),
            'risk_category': 'High' if probability > 0.45 else 'Medium' if probability > 0.3 else 'Low',
            'prediction': int(prediction),
            'top_risk_factors': [{'factor': factor, 'weight': round(weight, 4)} 
                               for factor, weight in top_factors],
            'model_confidence': round(max(probability, 1-probability), 4)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model_info', methods=['GET'])
def model_info():
    """Endpoint to get model metadata"""
    return jsonify({
        'model_type': 'Logistic Regression',
        'features': ['loan_amnt', 'term', 'int_rate',
       'emp_length', 'home_ownership', 'annual_inc',
       'verification_status', 'purpose','dti', 'open_acc', 'revol_bal',
       'revol_util', 'mort_acc', 'pub_rec_bankruptcies'],  # Update with your features
        'training_accuracy': 0.77,  # Add your actual metrics
        'precision': 0.45,
        'recall': 0.85,
        'f1_score': 0.6
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)