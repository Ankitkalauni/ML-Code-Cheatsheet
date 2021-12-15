!pip install scikit-learn-intelex --progress-bar off >> /tmp/pip_sklearnex.log

from sklearnex import patch_sklearn
patch_sklearn()


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import optuna
import pandas as pd, numpy as np, os
from sklearn.metrics import log_loss
from sklearn.svm import SVC

#change here
SEED = 1
FOLD = 5
train = X #train dataset
y = y #target dataset

def fit_lgb(trial, x_train, y_train, x_test, y_test):
    params = {
        'C': trial.suggest_loguniform('C', 1e-4, 1e4),
        'gamma': trial.suggest_loguniform('gamma', 1e-4, 1e4),
        'kernel': trial.suggest_categorical("kernel", ["linear", "rbf"])
    }
    
    model = SVC(**params)
    model.fit(x_train, y_train)
    
    y_train_pred = model.predict(x_train)
    
    y_test_pred = model.predict(x_test)
    y_train_pred = y_train_pred
    y_test_pred = y_test_pred
    
    log = {
        "loss": get_metrics(y_train, y_train_pred)}
    
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
  
  
  
  
study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=123),
                            direction="maximize",
                            pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=500, show_progress_bar=True, timeout= 8 * 60 * 60)


print(f"Best Value: {study.best_trial.value}")
print(f"Best Params: {study.best_params}")
