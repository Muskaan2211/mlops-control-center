import numpy as np
from scipy.stats import ks_2samp
from .config import FEATURES
from .data import load_data

def detect_drift(reference,current,alpha=.01):
    results={}
    for col in FEATURES:
        stat,p=ks_2samp(reference[col].dropna(),current[col].dropna())
        results[col]={'ks_statistic':float(stat),'p_value':float(p),'drifted':bool(p<alpha)}
    return {'features':results,'drifted_features':[k for k,v in results.items() if v['drifted']],
            'note':'KS p-values are uncorrected for multiple testing; inspect sample sizes.'}

if __name__=='__main__':
    import json
    reference=load_data()
    current=reference.sample(frac=.3,random_state=13).copy()
    current['monthly_charges']+=35
    print(json.dumps(detect_drift(reference,current),indent=2))
