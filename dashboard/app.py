import json
import os
from pathlib import Path
import requests
import streamlit as st
import plotly.express as px
import pandas as pd
from src.mlops.config import REPORT_PATH,DATA

st.set_page_config(page_title='MLOps Control Center',page_icon='🧠',layout='wide')
st.markdown("""<style>.block-container{padding-top:2rem} h1{font-weight:800} [data-testid="stMetric"]{background:#121a2c;border:1px solid #263a55;padding:18px;border-radius:16px}</style>""",unsafe_allow_html=True)
st.title('🧠 MLOps Control Center')
st.caption('MODEL OPERATIONS  /  TRAIN · EVALUATE · SERVE · MONITOR')
api=os.getenv('API_URL','http://localhost:8000')
try: health=requests.get(api+'/health',timeout=3).json()
except requests.RequestException: health={'status':'offline','model_ready':False}
status='🟢 Online' if health['status']=='ok' else '🔴 Offline'
st.sidebar.header('System status');st.sidebar.write(f'API: {status}')
st.sidebar.write('Model: '+('Ready' if health.get('model_ready') else 'Not trained'))
st.sidebar.info('Synthetic demo data. Metrics are not production results.')
report=json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}
metrics=report.get('metrics',{})
cols=st.columns(4)
for col,label,key in zip(cols,['ROC AUC','F1 score','Accuracy','Training trials'],['roc_auc','f1','accuracy','trials']):
    value=report.get('trials','—') if key=='trials' else (f'{metrics[key]:.3f}' if key in metrics else '—')
    col.metric(label,value)
tab1,tab2,tab3=st.tabs(['🔮 Live prediction','📊 Model performance','🛰️ Data monitoring'])
with tab1:
    st.subheader('Customer churn risk')
    with st.form('predict'):
        a,b,c=st.columns(3)
        tenure=a.slider('Tenure (months)',0,120,18); charges=b.slider('Monthly charges ($)',0.,200.,75.)
        tickets=c.slider('Support tickets',0,20,2); usage=a.slider('Usage hours',0.,250.,35.)
        contract=b.selectbox('Monthly contract',[0,1]); autopay=c.selectbox('Auto pay',[0,1]);satisfaction=a.slider('Satisfaction',1,5,3)
        submit=st.form_submit_button('Predict churn risk',type='primary')
    if submit:
        payload=dict(tenure_months=tenure,monthly_charges=charges,support_tickets=tickets,usage_hours=usage,contract_monthly=contract,auto_pay=autopay,satisfaction_score=satisfaction)
        try:
            r=requests.post(api+'/predict',json=payload,timeout=10);r.raise_for_status();out=r.json()
            st.metric('Estimated churn probability',f"{out['churn_probability']:.1%}")
            st.success('Likely to retain' if out['prediction']=='retain' else 'Higher churn risk')
        except requests.RequestException as e: st.error(f'Prediction failed: {e}')
with tab2:
    if not report: st.warning('Train a model to populate this panel.')
    else:
        st.json({'run_id':report['run_id'],'promoted':report['promoted'],'data_type':report['data_type'],'best_params':report['best_params']})
        fig=px.bar(x=list(metrics),y=list(metrics.values()),labels={'x':'Metric','y':'Score'},title='Held-out test metrics')
        st.plotly_chart(fig,use_container_width=True)
with tab3:
    st.subheader('Distribution shift detection')
    st.write('Simulate a change in customer charges and run a two-sample KS test.')
    if st.button('Run drift check'):
        from src.mlops.data import load_data
        from src.mlops.drift import detect_drift
        df=load_data(); shifted=df.sample(frac=.3,random_state=13).copy();shifted['monthly_charges']+=35
        result=detect_drift(df,shifted)
        st.write('Flagged features:',', '.join(result['drifted_features']) or 'None')
        st.dataframe(pd.DataFrame(result['features']).T,use_container_width=True)
        st.caption(result['note'])
