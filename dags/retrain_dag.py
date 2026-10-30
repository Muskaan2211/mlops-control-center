from datetime import datetime,timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG('mlops_churn_retraining',start_date=datetime(2025,1,1),schedule='@weekly',catchup=False,
         default_args={'retries':1,'retry_delay':timedelta(minutes=5)},tags=['mlops','churn']) as dag:
    retrain=BashOperator(task_id='train_validate_promote',bash_command='cd /opt/mlops && python -m src.mlops.training --trials 10')
