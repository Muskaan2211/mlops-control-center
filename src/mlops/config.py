import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
ARTIFACTS = ROOT / 'artifacts'
MODEL_PATH = ARTIFACTS / 'champion.joblib'
REPORT_PATH = ARTIFACTS / 'report.json'
MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', f'sqlite:///{ROOT / "mlflow.db"}')
FEATURES = ['tenure_months','monthly_charges','support_tickets','usage_hours','contract_monthly','auto_pay','satisfaction_score']
