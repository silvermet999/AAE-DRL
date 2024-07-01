import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import recall_score, accuracy_score

from sklearn.svm import SVC
import xgboost
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.utils import compute_sample_weight


import main
from skopt import BayesSearchCV, space



def gbc():
    clf = GradientBoostingClassifier(n_estimators=10, random_state=42)
    clf = clf.fit(main.x_train, main.y_train.iloc[:, 0])
    pred = clf.predict(main.x_test)
    class_report = classification_report(main.y_test.iloc[:10688, 0], pred)
    with open('gbc_classification_report.txt', 'w') as f:
        f.write(class_report)
    return pred, class_report





le = LabelEncoder()
y_train = le.fit_transform(main.y_train["Category"])
y_test = le.fit_transform(main.y_test["Category"])


# def xgb():
clf = xgboost.XGBClassifier(
    max_depth=16,
    gamma=.014,
    learning_rate=.02,
    reg_alpha=3,
    reg_lambda=.75,
    colsample_bytree=.64,
    min_child_weight=2,
    n_estimators=180,
    seed=0,
    objective='multi:softmax',
    num_class=14,
)

search_spaces = {
    'n_estimators': space.Integer(93, 181),
    'max_depth': space.Integer(9, 17),
    'learning_rate': space.Real(0.01, 0.24, 'log-uniform')
}


# opt = BayesSearchCV(
#     clf,
#     search_spaces=search_spaces,
#     n_iter=10,
#     random_state=0,
#     verbose=1,
#     n_jobs=-1,
#     scoring='accuracy',
#     cv=3
# )

y = main.y.iloc[:, 0]
sample_weights = np.ones_like(y, dtype=float)
sample_weights[3] = 2.0
sample_weights[9] = 2.0
sample_weights[13] = 3.0

kf = KFold(n_splits=3, shuffle=True, random_state=0)
sm = SMOTE(random_state=42)
scores = []

for train_index, val_index in kf.split(main.df):
    X_tr, X_val = main.X_rs[train_index], main.X_rs[val_index]
    y_tr, y_val = y.iloc[train_index], y.iloc[val_index]
    sample_weights_tr = sample_weights[train_index]

    X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
    original_sample_count = len(y_tr)
    synthetic_sample_count = len(y_tr_res) - original_sample_count
    synthetic_sample_weight = np.mean(sample_weights_tr)
    sample_weights_res = np.concatenate([
        sample_weights_tr,
        np.full(synthetic_sample_count, synthetic_sample_weight)
    ])

    clf.fit(X_tr_res, y_tr_res, sample_weight=sample_weights_res)
    y_pred = clf.predict(X_val)
    scores.append(accuracy_score(y_val, y_pred))

    class_report = classification_report(y_val[:10688], y_pred)

    with open('xgb_smote_classification_report.txt', 'w') as f:
        f.write(class_report)

    # return y_pred, class_report




adaboost_FT = {
    'n_estimators': hp.choice('n_estimators', np.arange(50, 500, dtype=int)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 1.0),
    'max_depth': hp.choice('max_depth', np.arange(1, 10, dtype=int)),
}


def adaboost():
    base_estimator = DecisionTreeClassifier(
        max_depth=250
    )
    clf = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=500,
        learning_rate=0.03,
        algorithm="SAMME",
        random_state=42
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(main.x_train):
        X_tr, X_val = main.x_train[train_index], main.x_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]
        sample_weights_tr = sample_weights[train_index]

        clf.fit(X_tr, y_tr, sample_weight=sample_weights_tr)
        y_pred = clf.predict(X_val)
        print(y_pred)
        # scores.append(recall_score(y_val, y_pred, average="macro"))
        class_report = classification_report(main.y_test.iloc[:4471, 0], y_pred)
        with open('adaboost_classification_tweek_report.txt', 'w') as f:
            f.write(class_report)


    return class_report

    #only category col
    # clf = clf.fit(main.x_train, main.y_train.iloc[:, 0])
    # pred = clf.predict(main.x_test)
    # class_report = classification_report(main.y_test.iloc[:10688, 0], pred)
    # with open('adaboost_classification_tweek_report.txt', 'w') as f:
    #     f.write(class_report)
    # return pred, class_report


# trials = Trials()
# best = fmin(fn=adaboost,
#             space=adaboost_FT,
#             algo=tpe.suggest,
#             max_evals=10,
#             trials=trials)


