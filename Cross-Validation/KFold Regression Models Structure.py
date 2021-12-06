models = [
 #('model_name_here', Model(**params))
]

# create dictionaries to store predictions
oof_pred_tmp = dict()
test_pred_tmp = dict()
scores_tmp = dict()


kf = KFold(n_splits=10,random_state=42,shuffle=True)

for fold, (train_indx,val_indx) in enumerate(kf.split(train)):

    X_train, X_valid = train.iloc[train_indx,:].reset_index(drop=True), train.iloc[val_indx,:].reset_index(drop=True)
    y_train, y_valid = y.iloc[train_indx].reset_index(drop=True), y.iloc[val_indx].reset_index(drop=True)
    
    # fit & predict all models on the same fold
    for name, model in models:
        if name not in scores_tmp:
            oof_pred_tmp[name] = list()
            oof_pred_tmp['y_valid'] = list()
            test_pred_tmp[name] = list()
            scores_tmp[name] = list()
        
        model.fit(X_train, y_train,
#                   eval_set=[(X_valid, y_valid)],
#            eval_metric="rmse",
#                  verbose=0,
#             early_stopping_rounds=1000,
#             use_best_model=True
                 )

    # validation prediction
        pred_valid = model.predict(X_valid)
        score = mean_squared_error(y_valid, pred_valid, squared=False)

        scores_tmp[name].append(score)
        oof_pred_tmp[name].extend(pred_valid)

        print(f"Fold: {fold + 1} Model: {name} Score: {score}")
        print('--'*20)

        # test prediction
        y_hat = model.predict(test)
        test_pred_tmp[name].append(y_hat)
        print('test prediction Done')

    # store y_validation for later use
    oof_pred_tmp['y_valid'].extend(y_valid)
        
# print overall validation scores
for name, model in models:
    print(f"Overall Validation Score | {name}: {np.mean(scores_tmp[name])}")
    print('::'*20)


############################################### BASE TEST PREDICTION ############################################### 

# create df with base predictions on test_data
base_test_predictions = pd.DataFrame(
    {name: np.mean(np.column_stack(test_pred_tmp[name]), axis=1) 
    for name in test_pred_tmp.keys()}
)


# save csv checkpoint
base_test_predictions.to_csv('./base_test_predictions.csv', index=False)

# create simple average blend 
base_test_predictions['simple_avg'] = base_test_predictions.mean(axis=1)

# create submission file with simple blend average
simple_blend_submission = sample_submission.copy()
simple_blend_submission['Sales'] = base_test_predictions['simple_avg']
simple_blend_submission.to_csv('./simple_blend_submission.csv', index=False)


############################################### OOF PREDICTION ###############################################

# create training set for meta learner based on the oof_predictions of the base models
oof_predictions = pd.DataFrame(
    {name:oof_pred_tmp[name] for name in oof_pred_tmp.keys()}
)

# save csv checkpoint
oof_predictions.to_csv('./oof_predictions.csv', index=False)

# get simple blend validation score
y_valid = oof_predictions['y_valid'].copy()
y_hat_blend = oof_predictions.drop(columns=['y_valid']).mean(axis=1)
score = mean_squared_error(y_valid, y_hat_blend,squared=False)

print(f"Overall Validation Score | Simple Blend: {score}")
print('::'*20)
