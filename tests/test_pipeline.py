import numpy as np
from fastapi.testclient import TestClient
from src.mlops.data import generate_data,split_data
from src.mlops.drift import detect_drift
from api.main import app

def test_data_reproducible():
    assert generate_data(50,7).equals(generate_data(50,7))

def test_split_disjoint_sizes():
    df=generate_data(100)
    xtr,xte,ytr,yte=split_data(df)
    assert len(xtr)==80 and len(xte)==20

def test_drift_flags_large_shift():
    df=generate_data(1000)
    shifted=df.copy();shifted['monthly_charges']+=100
    assert 'monthly_charges' in detect_drift(df,shifted)['drifted_features']

def test_health():
    response=TestClient(app).get('/health')
    assert response.status_code==200

def test_invalid_payload():
    response=TestClient(app).post('/predict',json={'tenure_months':-1})
    assert response.status_code==422
