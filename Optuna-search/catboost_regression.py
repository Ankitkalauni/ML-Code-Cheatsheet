import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, StratifiedKFold

# create trial function
# OPTUNA_OPTIMIZATION = True

def objective(trial, X=X, y=y):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    params = {
        'iterations':trial.suggest_int("iterations", 10000, 20000),
        'objective': trial.suggest_categorical('objective', ['RMSE']),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
        'learning_rate' : trial.suggest_uniform('learning_rate',0.001,0.01),
        'reg_lambda': trial.suggest_uniform('reg_lambda',1e-5,100),
        'random_strength': trial.suggest_uniform('random_strength',10,50),
        'depth': trial.suggest_int('depth',1,15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,30),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,15),
        'verbose': False,
#         'task_type' : 'GPU',
#         'devices' : '0'
    }
    
    params['grow_policy'] = 'Depthwise'
    params['iterations'] = 10000
    
    if params['bootstrap_type'] == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
    elif params['bootstrap_type'] == 'Bernoulli':
        params['subsample'] = trial.suggest_float('subsample', 0.1, 1)
    
    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test,y_test)],
        early_stopping_rounds=100,
        use_best_model=True
    )
    
    # validation prediction
    y_hat = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_hat,squared=False)
    score = rmse
    
    del X_train, y_train, X_test, y_test, y_hat
    gc.collect()
    return rmse
