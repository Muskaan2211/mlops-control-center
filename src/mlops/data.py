import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DATA, FEATURES

def generate_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'tenure_months': rng.integers(1,73,n),
        'monthly_charges': rng.uniform(20,130,n).round(2),
        'support_tickets': rng.poisson(1.7,n),
        'usage_hours': rng.gamma(3,9,n).round(2),
        'contract_monthly': rng.integers(0,2,n),
        'auto_pay': rng.integers(0,2,n),
        'satisfaction_score': rng.integers(1,6,n),
    })
    logit = (-0.6 - .028*df.tenure_months + .016*(df.monthly_charges-65)
             + .31*df.support_tickets - .014*df.usage_hours
             + .8*df.contract_monthly - .5*df.auto_pay - .36*(df.satisfaction_score-3))
    prob = 1/(1+np.exp(-logit))
    df['churn'] = rng.binomial(1,prob)
    return df

def load_data(path=None):
    if path:
        df = pd.read_csv(path)
    else:
        DATA.mkdir(exist_ok=True,parents=True)
        csv = DATA/'synthetic_customers.csv'
        if not csv.exists(): generate_data().to_csv(csv,index=False)
        df = pd.read_csv(csv)
    missing = set(FEATURES+['churn'])-set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    if df[FEATURES+['churn']].isna().any().any(): raise ValueError('Missing values found')
    if not set(df.churn.unique()).issubset({0,1}): raise ValueError('churn must be binary')
    return df

def split_data(df):
    x,y=df[FEATURES],df.churn.astype(int)
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.2,random_state=42,stratify=y)
    return xtrain,xtest,ytrain,ytest
