from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import optuna
import pandas as pd, numpy as np, os
from sklearn.metrics import log_loss

#change here
SEED = 1
FOLD = 5
train = X #train dataset
y = y #target dataset

def fit_lgb(trial, x_train, y_train, x_test, y_test):
    param_grid = {'C': trial.suggest_loguniform("C", 1e-5, 1e2),
                  'max_iter': trial.suggest_int('max_iter',20,1000),
                   'solver': trial.suggest_categorical('solver',['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
                 }
    if param_grid['solver'] == 'saga':
        param_grid['penalty'] : trial.suggest_categorical('penalty',['l1', 'l2', 'elasticnet', 'none'])
        param_grid['l1_ratio'] : trial.suggest_uniform('l1_ratio',0.0,1.0)
    elif param_grid['solver'] == 'sag':
        param_grid['penalty'] : trial.suggest_categorical('penalty',['l2' 'none'])
            
    elif param_grid['solver'] == 'liblinear':
        param_grid['penalty'] : trial.suggest_categorical('penalty',['l1', 'l2'])
            
    elif param_grid['solver'] == 'newton-cg':
        param_grid['penalty'] : trial.suggest_categorical('penalty',[ 'l2' 'none'])
    
    elif param_grid['solver'] == 'lbfgs':
        param_grid['penalty'] : trial.suggest_categorical('penalty',[ 'l2' 'none'])
    
    model = LogisticRegression(**param_grid,n_jobs=-1)
    model.fit(x_train, y_train)
    
    y_train_pred = model.predict_proba(x_train)[:,1]
    
    y_test_pred = model.predict_proba(x_test)[:,1]
    y_train_pred = y_train_pred
    y_test_pred = y_test_pred
    
    log = {
        "loss": log_loss(y_train, y_train_pred)}
    
    return model, log

def objective(trial):
    log = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    for fold, (idx_train, idx_valid) in enumerate(kf.split(X, y)):
        # create train, validation sets
        X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
        X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]

#     x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.2)
        model, log = fit_lgb(trial, X_train, y_train, X_valid, y_valid)
    log += log['loss']
        
    return log

study = optuna.create_study(direction="minimize", study_name="Logistic Regression")
study.optimize(objective, n_trials=100)