import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import KFold, GridSearchCV
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score, auc, roc_curve, recall_score
from matplotlib.legend_handler import HandlerLine2D
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.svm import SVC
import xgboost
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, ADASYN
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



def xgb():
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

    # search_spaces = {
    #     'n_estimators': space.Integer(93, 181),
    #     'max_depth': space.Integer(9, 17),
    #     'learning_rate': space.Real(0.01, 0.24, 'log-uniform')
    # }


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

    le = LabelEncoder()
    y = le.fit_transform(main.y["Category"])
    sample_weights = np.ones_like(y, dtype=float)
    sample_weights[3] = 2
    sample_weights[9] = 2.0
    sample_weights[13] = 3.0

    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    sm = SMOTE(random_state=42)
    scores = []

    for train_index, val_index in kf.split(main.df):
        X_tr, X_val = main.X_rs[train_index], main.X_rs[val_index]
        y_tr, y_val = y[train_index], y[val_index]
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

        class_report = classification_report(y_val, y_pred)

        with open('xgb/xgb_kf_smote_sing_tweek_classification_report.txt', 'w') as f:
            f.write(class_report)

    return y_pred, class_report





param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'estimator__max_depth': [1, 2, 3]
}


def adaboost():
    base_estimator = DecisionTreeClassifier(
        max_depth=50
    )
    clf = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=50,
        learning_rate=0.1,
        algorithm="SAMME",
        random_state=42
    )
    y = main.y.iloc[:, 0]
    sample_weights = np.ones_like(y, dtype=float)
    sample_weights[3] = 2.0
    sample_weights[9] = 2.0
    sample_weights[13] = 3.0

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(main.x_train):
        X_tr, X_val = main.x_train[train_index], main.x_train[val_index]
        y_tr, y_val = y[train_index], y[val_index]
        sample_weights_tr = sample_weights[train_index]

        clf.fit(X_tr, y_tr, sample_weight=sample_weights_tr)
        y_pred = clf.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred))
        class_report = classification_report(y_val, y_pred)
        with open('adaboost/adaboost_classification_t_report.txt', 'w') as f:
            f.write(class_report)


    return class_report



def objective(model, param):
    trials = Trials()
    best = fmin(fn=model,
                space=param,
                algo=tpe.suggest,
                max_evals=10,
                trials=trials)
    return best


# ada = AdaBoostClassifier(estimator=base_estimator)
#
# grid_search = GridSearchCV(ada, param_grid=param_grid, cv=5, scoring='accuracy')
# grid_search.fit(main.x_train, main.y_train.iloc[:, 0])
#
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_
#
# print("Best parameters:", best_params)
# print("Best cross-validation score:", best_score)

le = LabelEncoder()
y_train_bin = le.fit_transform(main.y_train["Category"])
y_test_bin = le.fit_transform(main.y_test["Category"])

space = {
    "learning_rate": [0.25, 0.1, 0.05, 0.01],
    "n_estimators": [8, 16, 32, 64, 100, 200],
    "max_depth": list(range(1, 33)),
    "min_samples_split": np.arange(0.1, 1.1, 0.1),
    "min_samples_leaf": np.arange(0.1, 0.6, 0.1),
    "max_features": list(range(1, main.x_train.shape[1] + 1))
}


def gbc():
    clf = GradientBoostingClassifier(
        max_depth=16,
        n_estimators=100,
        learning_rate=.01,
        min_samples_leaf=10,
        max_features="sqrt",
        min_samples_split=10,
    )
    le = LabelEncoder()
    y = le.fit_transform(main.y["Category"])
    sample_weights = np.ones_like(y, dtype=float)
    sample_weights[3] = 2.0
    sample_weights[9] = 2.0
    sample_weights[13] = 3.0

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(main.x_train):
        X_tr, X_val = main.x_train[train_index], main.x_train[val_index]
        y_tr, y_val = y[train_index], y[val_index]
        sample_weights_tr = sample_weights[train_index]

        clf.fit(X_tr, y_tr, sample_weight=sample_weights_tr)
        y_pred = clf.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred))
        class_report = classification_report(y_val, y_pred)
        with open('gbc_classification_report.txt', 'w') as f:
            f.write(class_report)
    return class_report


# gradientboosting = GradientBoostingClassifier()
#
# grid_search = GridSearchCV(gradientboosting, param_grid=space, cv=5, scoring='accuracy')
# grid_search.fit(main.x_train, main.y_train["Category"])
#
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_
#
# print("Best parameters:", best_params)
# print("Best cross-validation score:", best_score)

