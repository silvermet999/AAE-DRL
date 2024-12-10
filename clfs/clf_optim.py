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
    X_train = pd.DataFrame(np.loadtxt('/AAE/Adam.txt'))
    # X_train = pd.DataFrame(main_u.X_train_sc)
    y_train = classifier.y

    # classifier_name = trial.suggest_categorical("classifier", ["SVC", "rf"])
    # if classifier_name == "SVC":
    #     svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
    #     classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")

    #
    # elif classifier_name == "gb":
    #     gb_n_estimators = trial.suggest_int("gb_n_estimators", 50, 200)
    #     gb_learning_rate = trial.suggest_float("gb_learning_rate", 1e-3, 1e-1, log=True)
    #     gb_max_depth = trial.suggest_int("gb_max_depth", 3, 12)
    #     classifier_obj = sklearn.ensemble.GradientBoostingClassifier(
    #         n_estimators=gb_n_estimators,
    #         learning_rate=gb_learning_rate,
    #         max_depth=gb_max_depth
    #     )

    k = trial.suggest_int('n_neighbors', 1, 26)
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
    p = trial.suggest_int('p', 1, 3) if metric == 'minkowski' else 2
    leaf_size = trial.suggest_int("leaf_size", 10, 80)
    classifier_obj = KNeighborsClassifier(
        n_neighbors=k, metric=metric, p=p, leaf_size=leaf_size
    )



    # rf_max_depth = trial.suggest_int("rf_max_depth", 10, 20)
    # rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 200)
    # classifier_obj = sklearn.ensemble.RandomForestClassifier(
    #     max_depth=rf_max_depth,
    #     n_estimators=rf_n_estimators,
    # )

    recall_scorer = sklearn.metrics.make_scorer(sklearn.metrics.recall_score, average='weighted')
    score = sklearn.model_selection.cross_val_score(classifier_obj, X_train, y_train, cv=5, scoring=recall_scorer).mean()
    return score

X = main_u.inverse_sc(main_u.X.to_numpy(), np.loadtxt('/home/silver/PycharmProjects/AAEDRL/AAE/smp2.txt'))
# X_train = main_u.inverse_sc(main_u.X_train.to_numpy(), X)
# X_test = main_u.inverse_sc(main_u.X_test.to_numpy(), main_u.X_test_sc)
X_train, X_test, y_train, y_test = main_u.vertical_split(X, main_u.df["attack_cat"])
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

def objectivexgb(trial):
    param = {
        "verbosity": 0,
        "objective": "multi:softmax",
        "num_class": 22,
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
study.optimize(objectivexgb, n_trials=200)
print(study.best_trial)
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_params = study.best_params


# [I 2024-12-06 22:22:56,284] Trial 85 finished with value: 0.9749032882011606 and parameters:
# {'booster': 'gbtree', 'lambda': 1.955696877876371e-05, 'alpha': 3.4821998334693364e-13,
# 'subsample': 0.9640929261558455, 'colsample_bytree': 0.9027725034365305, 'max_depth': 24,
# 'min_child_weight': 7, 'eta': 1.4433639109132085e-08, 'gamma': 0.0009949289920304545,
# 'grow_policy': 'depthwise'}. Best is trial 85 with value: 0.9749032882011606.
