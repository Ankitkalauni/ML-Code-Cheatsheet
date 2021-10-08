import optuna
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold, StratifiedKFold

# create trial function
OPTUNA_OPTIMIZATION = True

def objective(trial):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)
    
    params = {
        'iterations':trial.suggest_int("iterations", 1000, 20000),
        'objective': trial.suggest_categorical('objective', ['Logloss', 'CrossEntropy']),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
        'od_wait':trial.suggest_int('od_wait', 500, 2000),
        'learning_rate' : trial.suggest_uniform('learning_rate',0.02,1),
        'reg_lambda': trial.suggest_uniform('reg_lambda',1e-5,100),
        'random_strength': trial.suggest_uniform('random_strength',10,50),
        'depth': trial.suggest_int('depth',1,15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,30),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,15),
        'verbose': True,
        'task_type' : 'GPU',
        'devices' : '0'
    }
    
    if params['bootstrap_type'] == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
    elif params['bootstrap_type'] == 'Bernoulli':
        params['subsample'] = trial.suggest_float('subsample', 0.1, 1)
    
    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test,y_test)],
        early_stopping_rounds=100,
        use_best_model=True
    )
    
    # validation prediction
    y_hat = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_hat)
    score = auc(fpr, tpr)
    
    return score

study = optuna.create_study(direction="minimize", study_name="LGBM")
study.optimize(objective, n_trials=100,timeout = STUDY_TIME)

study.best_params