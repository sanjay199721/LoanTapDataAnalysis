import pandas as pd
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