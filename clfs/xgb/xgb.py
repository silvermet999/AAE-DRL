import numpy as np
from sklearn.model_selection import KFold, train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder
import xgboost
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from AAE import main
# import torch
# from clfs import new_dataset

# cuda = True if torch.cuda.is_available() else False

X_train = main.X_train_rs
y_train = main.y_train
X_test = main.X_test_rs
y_test = main.y_test

# [I 2024-08-13 13:22:54,103] Trial 85 finished with value: 0.9707185941900267 and parameters: {'classifier': 'xgb', 'xgb_max_depth': 12, 'xgb_learning_rate': 0.06459585378973996, 'xgb_n_estimators': 186}. Best is trial 85 with value: 0.9707185941900267.

params = {"booster": "dart", "lambda": 0.027899343813320206, "alpha": 6.645419741159056e-08, "subsample": 0.7970952489523347, "colsample_bytree": 0.6529288715068651, "max_depth": 44, "min_child_weight": 5, "eta": 0.6946846733929531, "gamma": 0.40023014808557217, "grow_policy": "depthwise", "sample_type": "weighted", "normalize_type": "forest", "rate_drop": 0.9757015105398882, "skip_drop": 2.86041604442057e-06}


# def xgb():
# params = xgboost.XGBClassifier(
#     max_depth= 16,
#     gamma=params.014,
#     # learning_rate=params[], #0.2
#     reg_alpha=params["alpha"], #3
#     reg_lambda=params["lambda"], #.75
#     colsample_bytree=params["colsample_bytree"], #.64
#     min_child_weight=params["min_child_weight"], #2
#     # n_estimators=, #180
#     # seed=0,
#     objective='binary:logistic', #multi:softmax
#     booster = params["booster"],
#     subsample = params["subsample"],
#     eta = params["eta"],
#     sample_type = params["sample_type"],
#     normalize_type = params["normalize_type"],
#     rate_drop = params["rate_drop"],
#     grow_policy = params["grow_policy"],
#
# )

dtrain = xgboost.DMatrix(X_train, label=y_train)
dvalid = xgboost.DMatrix(X_test, label=y_test)
bst = xgboost.train(params, dtrain)
preds = bst.predict(dvalid)
pred_labels = np.rint(preds)
# pred_labels = np.argmax(preds, axis=1)
class_report = classification_report(y_test, pred_labels)

    # le = LabelEncoder()
    # y = le.fit_transform(main.y["Category"])
    # sample_weights = np.ones_like(y, dtype=float)
    # sample_weights[3] = 2
    # sample_weights[9] = 2.0
    # sample_weights[13] = 3.0
    #
    # kf = KFold(n_splits=10, shuffle=True, random_state=0)
    # sm = SMOTE(random_state=42)
    # scores = []
    #
    # for train_index, val_index in kf.split(main.df):
    #     X_tr, X_val = df.iloc[train_index], df.iloc[val_index]
    #     y_tr, y_val = y[train_index], y[val_index]
    #     sample_weights_tr = sample_weights[train_index]
    #
    #     X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
    #     original_sample_count = len(y_tr)
    #     synthetic_sample_count = len(y_tr_res) - original_sample_count
    #     synthetic_sample_weight = np.mean(sample_weights_tr)
    #     sample_weights_res = np.concatenate([
    #         sample_weights_tr,
    #         np.full(synthetic_sample_count, synthetic_sample_weight)
    #     ])
    #
    #     clf.fit(X_tr_res, y_tr_res, sample_weight=sample_weights_res)
    #     y_pred = clf.predict(X_val)
    #     scores.append(accuracy_score(y_val, y_pred))
    #
    #     class_report = classification_report(y_val, y_pred)
    #
    #     with open('xgb_kf10_smote_synth_sup_classification_report.txt', 'w') as f:
    #         f.write(class_report)

    # return y_pred, class_report


# space = {
#     'max_depth': hp.quniform('max_depth', 14, 30, 1),
#     'gamma': hp.uniform('gamma', 0.1, 1),
#     'learning_rate': hp.uniform('learning_rate', np.log(0.01), np.log(0.1)),
#     'reg_alpha': hp.quniform('reg_alpha', 90, 200, 1),
#     'reg_lambda': hp.uniform('reg_lambda', 0, 1),
#     'colsample_bytree': hp.uniform('colsample_bytree', 0.4, 0.9),
#     'min_child_weight': hp.quniform('min_child_weight', 10, 40, 1),
#     'max_delta_step' : hp.uniform('max_delta_step', 0, 5)
# }
# def objective(space):
#     clf=xgboost.XGBClassifier(
#                     n_estimators =180, max_depth = int(space['max_depth']), gamma = space['gamma'],
#                     reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
#                     colsample_bytree=int(space['colsample_bytree']), learning_rate=space['learning_rate'],
#                                          device='cuda')
#
#     le = LabelEncoder()
#     y = le.fit_transform(main.y["Category"])
#
#     X_train, X_test, y_train, y_test = train_test_split(main.df[:39744], y, test_size=.2)
#
#     evaluation = [(X_train, y_train), (X_test, y_test)]
#
#     clf.fit(X_train, y_train,
#             eval_set=evaluation)
#
#     pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, pred)
#     return {'loss': -accuracy, 'status': STATUS_OK }
# trials = Trials()
# best = fmin(fn=objective,
#             space=space,
#             algo=tpe.suggest,
#             max_evals=100,
#             trials=trials,
#             verbose=2)
# print("Best hyperparameters:", best)
