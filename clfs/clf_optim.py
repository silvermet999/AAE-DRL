import numpy as np
import optuna
import pandas as pd

import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier

from data import main_u
import sklearn.metrics




def objective(trial):

    gb_n_estimators = trial.suggest_int("gb_n_estimators", 10, 30)
    gb_learning_rate = trial.suggest_float("gb_learning_rate", 1e-3, 1e-1, log=True)
    gb_max_depth = trial.suggest_int("gb_max_depth", 3, 12)
    classifier_obj = sklearn.ensemble.GradientBoostingClassifier(
        n_estimators=gb_n_estimators,
        learning_rate=gb_learning_rate,
        max_depth=gb_max_depth
    )

    # k = trial.suggest_int('n_neighbors', 1, 30)
    # metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
    # p = trial.suggest_int('p', 1, 3) if metric == 'minkowski' else 2
    # leaf_size = trial.suggest_int("leaf_size", 10, 80)
    # classifier_obj = KNeighborsClassifier(
    #     n_neighbors=k, metric=metric, p=p, leaf_size=leaf_size
    # )



    # rf_max_depth = trial.suggest_int("rf_max_depth", 10, 20)
    # rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 200)
    # classifier_obj = sklearn.ensemble.RandomForestClassifier(
    #     max_depth=rf_max_depth,
    #     n_estimators=rf_n_estimators,
    # )

    recall_scorer = sklearn.metrics.make_scorer(sklearn.metrics.recall_score, average='weighted')
    score = sklearn.model_selection.cross_val_score(classifier_obj, X_train, y_train, cv=5, scoring=recall_scorer).mean()
    return score

df = pd.DataFrame(pd.read_csv("/home/silver/PycharmProjects/AAEDRL/AAE/ds_synth.csv"))[:141649]
df_disc, df_cont = main_u.df_type_split(df)
_, mainX_cont = main_u.df_type_split(main_u.X)
X_inv = main_u.inverse_sc_cont(mainX_cont, df_cont)
X = df_disc.join(X_inv)

X_train, X_test, y_train, y_test = main_u.vertical_split(X, main_u.y)
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

def objectivexgb(trial):
    param = {
        "verbosity": 0,
        "objective": "multi:softmax",
        "num_class": 30,
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
        param["max_depth"] = trial.suggest_int("max_depth", 1, 50)
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

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    recall = sklearn.metrics.recall_score(y_test, pred_labels, average = "weighted")
    return recall

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)
print(study.best_trial)
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_params = study.best_params

# booster': 'dart', 'lambda': 1.2741189898147751e-06, 'alpha': 1.4321357588867728e-08, 'subsample': 0.8562739949422293, 'colsample_bytree': 0.5377392681680526, 'max_depth': 14, 'min_child_weight': 2, 'eta': 0.15241836486978852, 'gamma': 0.0006231718908289008, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'forest', 'rate_drop': 7.448781837495591e-06, 'skip_drop': 2.1073884076102898e-10
# [I 2025-01-23 08:52:23,884] Trial 18 finished with value: 0.9378215593698981 and parameters: {'gb_n_estimators': 28, 'gb_learning_rate': 0.06226465448010815, 'gb_max_depth': 9}. Best is trial 18 with value: 0.9378215593698981.
