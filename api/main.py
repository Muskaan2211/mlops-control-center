import json
from functools import lru_cache
from pathlib import Path
import joblib
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from prometheus_client import Counter,Histogram,make_asgi_app
from src.mlops.config import MODEL_PATH,REPORT_PATH,FEATURES

app=FastAPI(title='MLOps Control Center',version='1.0.0',description='Churn prediction with versioned local champion')
app.mount('/metrics',make_asgi_app())
PREDICTIONS=Counter('churn_predictions_total','Total predictions', ['label'])
LATENCY=Histogram('churn_prediction_seconds','Prediction duration')
class Customer(BaseModel):
    tenure_months:int=Field(ge=0,le=120)
    monthly_charges:float=Field(ge=0,le=1000)
    support_tickets:int=Field(ge=0,le=100)
    usage_hours:float=Field(ge=0,le=10000)
    contract_monthly:int=Field(ge=0,le=1)
    auto_pay:int=Field(ge=0,le=1)
    satisfaction_score:int=Field(ge=1,le=5)

@lru_cache(maxsize=2)
def get_model(path,mtime):
    return joblib.load(path)

@app.get('/health')
def health():
    return {'status':'ok','model_ready':MODEL_PATH.exists()}

@app.get('/model/info')
def model_info():
    if not REPORT_PATH.exists(): raise HTTPException(503,'Train a model first')
    return json.loads(REPORT_PATH.read_text())

@app.post('/predict')
def predict(customer:Customer):
    if not MODEL_PATH.exists(): raise HTTPException(503,'Model unavailable; run training first')
    import pandas as pd
    with LATENCY.time():
        model=get_model(str(MODEL_PATH),MODEL_PATH.stat().st_mtime_ns)
        row=pd.DataFrame([customer.model_dump()])[FEATURES]
        prob=float(model.predict_proba(row)[0,1])
    label='churn' if prob>=.5 else 'retain'
    PREDICTIONS.labels(label=label).inc()
    return {'churn_probability':round(prob,5),'prediction':label,'threshold':0.5}
