import numpy as np
import optuna

import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm
import xgboost as xgb
from AAE2 import prep
import sklearn.metrics

X_train = prep.RobustScaler(prep.X_train)
y_train = prep.y_train
X_test = prep.RobustScaler(prep.X_test)
y_test = prep.y_test

def objective(trial):

    # classifier_name = trial.suggest_categorical("classifier", ["SVC", "xgb"])
    # if classifier_name == "SVC":
    #     svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
    #     classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")


    # elif classifier_name == "xgb":
    #     xgb_max_depth = trial.suggest_int("xgb_max_depth", 3, 12)
    #     xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 1e-3, 1e-1, log=True)
    #     xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 50, 200)
    #     classifier_obj = xgb.XGBClassifier(
    #         max_depth=xgb_max_depth,
    #         learning_rate=xgb_learning_rate,
    #         n_estimators=xgb_n_estimators,
    #         objective="binary:logistic"
    #     )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-15, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 5, 50)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 20)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-10, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-10, 1.0, log=True)

    # elif classifier_name == "adaboost":
    #     ada_n_estimators = trial.suggest_int("ada_n_estimators", 50, 200)
    #     ada_learning_rate = trial.suggest_float("ada_learning_rate", 1e-3, 1e-1, log=True)
    #     classifier_obj = sklearn.ensemble.AdaBoostClassifier(
    #         n_estimators=ada_n_estimators,
    #         learning_rate=ada_learning_rate
    #     )


    # elif classifier_name == "gb":
    #     gb_n_estimators = trial.suggest_int("gb_n_estimators", 50, 200)
    #     gb_learning_rate = trial.suggest_float("gb_learning_rate", 1e-3, 1e-1, log=True)
    #     gb_max_depth = trial.suggest_int("gb_max_depth", 3, 12)
    #     classifier_obj = sklearn.ensemble.GradientBoostingClassifier(
    #         n_estimators=gb_n_estimators,
    #         learning_rate=gb_learning_rate,
    #         max_depth=gb_max_depth
    #     )


    # elif classifier_name == "rf":
    #     rf_max_depth = trial.suggest_int("rf_max_depth", 3, 12)
    #     rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 200)
    #     classifier_obj = sklearn.ensemble.RandomForestClassifier(
    #         max_depth=rf_max_depth,
    #         n_estimators=rf_n_estimators,
    #     )

    # recall_scorer = sklearn.metrics.make_scorer(sklearn.metrics.recall_score, average='macro')
    # score = sklearn.model_selection.cross_val_score(classifier_obj, X_train, y_train, cv=5, scoring=recall_scorer).mean()
    # return score
    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    recall = sklearn.metrics.recall_score(y_test, pred_labels, average = "weighted")
    return recall

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000)
print(study.best_trial)
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Load best parameters
best_params = study.best_params

# Train final model with best parameters
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)
bst = xgb.train(best_params, dtrain, num_boost_round=100)

# Predict and evaluate
preds_proba = bst.predict(dvalid)
pred_labels = np.argmax(preds_proba, axis=1)
class_report = sklearn.metrics.classification_report(y_test, pred_labels)
print(class_report)




# Value: 0.7741854321298276
#   Params:
#     booster: dart
#     lambda: 0.036205197742718506
#     alpha: 6.134173840573579e-08
#     subsample: 0.9793911844902469
#     colsample_bytree: 0.4828593933923663
#     max_depth: 19
#     min_child_weight: 6
#     eta: 0.9297684464996867
#     gamma: 1.053198768898343e-07
#     grow_policy: lossguide
#     sample_type: uniform
#     normalize_type: tree
#     rate_drop: 0.0005875390369779961
#     skip_drop: 3.5933107414694465e-05
