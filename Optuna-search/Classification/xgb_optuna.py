from optuna.integration import XGBoostPruningCallback
import lightgbm as lgbm

def objective(trial, X=X, y=y):
    # XGBoost parameters
    params = {
        "verbose": 0,  # 0 (silent) - 3 (debug)
        "objective": "binary:logistic",
        "n_estimators": 10000,
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
        "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 0.6),
        "subsample": trial.suggest_loguniform("subsample", 0.4, 0.8),
        "alpha": trial.suggest_loguniform("alpha", 0.01, 10.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "gamma": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 10, 1000),
        "seed": SEED,
        "n_jobs": -1,
        'tree_method':'gpu_hist',
        'gpu_id':0,
        
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(**params)
        pruning_callback = XGBoostPruningCallback(trial, "validation_0-logloss")
        model.fit(
            X_train, y_train,
        eval_set=[(X_test,y_test)],
        early_stopping_rounds=100,verbose=0)
        preds = model.predict_proba(X_test)
        cv_scores[idx] = log_loss(y_test, preds)

    return np.mean(cv_scores)

study = optuna.create_study(direction="minimize", study_name="XGBM")
study.optimize(objective, n_trials=100,timeout = STUDY_TIME)

study.best_params