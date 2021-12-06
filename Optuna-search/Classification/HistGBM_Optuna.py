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
    params = {
            'l2_regularization': trial.suggest_loguniform('l2_regularization',1e-10,10.0),
            'early_stopping': trial.suggest_categorical('early_stopping', ['True']),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001,0.1),
            'max_iter': trial.suggest_categorical('max_iter', [10000]),
            'max_depth': trial.suggest_int('max_depth', 2,30),
            'max_bins': trial.suggest_int('max_bins', 100,255),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20,100000),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20,80),
        }
    
    model = HistGradientBoostingClassifier(**params,verbose=1000)
    model.fit(x_train, y_train)
        
    y_test_pred = model.predict_proba(x_test)[:,1]
    log = {
        "loss": log_loss(y_test, y_test_pred)}
    
    return model, log

def objective(trial):
    _log = 0.0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    for fold, (idx_train, idx_valid) in enumerate(kf.split(X, y)):
        # create train, validation sets
        X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
        X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]

#     x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.2)
        model, log = fit_lgb(trial, X_train, y_train, X_valid, y_valid)
    _log += log['loss']
        
    return _log

study = optuna.create_study(direction="minimize", study_name="Histgbm Classifier")
study.optimize(objective, n_trials=25,timeout=STUDY_TIME)

study.best_trial.params