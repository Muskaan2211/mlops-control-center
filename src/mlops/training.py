import argparse
import json
import os
import tempfile
from datetime import datetime,timezone
import joblib
import mlflow
import numpy as np
import optuna
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from .config import ARTIFACTS, MODEL_PATH, REPORT_PATH, MLFLOW_URI, FEATURES
from .data import load_data, split_data

def make_model(params):
    return Pipeline([('imputer',SimpleImputer(strategy='median')),('model',XGBClassifier(
        n_estimators=params['n_estimators'],max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        objective='binary:logistic',eval_metric='logloss',random_state=42,n_jobs=2))])

def metrics(model,x,y):
    proba=model.predict_proba(x)[:,1]
    pred=(proba>=.5).astype(int)
    return { 'roc_auc':float(roc_auc_score(y,proba)), 'f1':float(f1_score(y,pred,zero_division=0)),
        'accuracy':float(accuracy_score(y,pred)), 'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0))}

def train(trials=10,data_path=None,force=False):
    df=load_data(data_path)
    xtrain,xtest,ytrain,ytest=split_data(df)
    # Isolate validation from the final held-out test set.
    from sklearn.model_selection import train_test_split
    xfit,xval,yfit,yval=train_test_split(xtrain,ytrain,test_size=.25,stratify=ytrain,random_state=43)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment('churn-optimization')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        params={'n_estimators':trial.suggest_int('n_estimators',60,200),
                'max_depth':trial.suggest_int('max_depth',2,7),
                'learning_rate':trial.suggest_float('learning_rate',.015,.25,log=True),
                'subsample':trial.suggest_float('subsample',.65,1),
                'colsample_bytree':trial.suggest_float('colsample_bytree',.65,1)}
        with mlflow.start_run(run_name=f'trial-{trial.number}'):
            model=make_model(params);model.fit(xfit,yfit)
            score=metrics(model,xval,yval)
            mlflow.log_params(params);mlflow.log_metrics({f'val_{k}':v for k,v in score.items()})
        return score['roc_auc']
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective,n_trials=trials)
    model=make_model(study.best_params);model.fit(xtrain,ytrain)
    test_metrics=metrics(model,xtest,ytest)
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    with mlflow.start_run(run_name='candidate-evaluation') as run:
        mlflow.log_params(study.best_params)
        mlflow.log_metrics({f'test_{k}':v for k,v in test_metrics.items()})
        with tempfile.TemporaryDirectory() as tmp:
            path=os.path.join(tmp,'model.joblib');joblib.dump(model,path)
            mlflow.log_artifact(path,artifact_path='model')
        run_id=run.info.run_id
    old_auc=-1
    if REPORT_PATH.exists():
        try: old_auc=json.loads(REPORT_PATH.read_text())['metrics']['roc_auc']
        except (KeyError,ValueError): pass
    promoted=bool(force or not MODEL_PATH.exists() or test_metrics['roc_auc']>=old_auc)
    report={'timestamp':datetime.now(timezone.utc).isoformat(),'metrics':test_metrics,
        'best_params':study.best_params,'trials':trials,'run_id':run_id,
        'promoted':promoted,'dataset_rows':len(df),'data_type':'synthetic' if not data_path else 'user_supplied',
        'baseline_auc':old_auc if old_auc>=0 else None}
    if promoted:
        # Atomic replacement: a concurrent API read never sees a partially written file.
        tmp=ARTIFACTS/'candidate.tmp';joblib.dump(model,tmp);os.replace(tmp,MODEL_PATH)
        REPORT_PATH.write_text(json.dumps(report,indent=2))
    (ARTIFACTS/'latest_candidate.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--trials',type=int,default=10)
    parser.add_argument('--data',default=None)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    print(json.dumps(train(args.trials,args.data,args.force),indent=2))
