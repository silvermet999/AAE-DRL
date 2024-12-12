import numpy as np

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_predict
from data import main_u
from sklearn.neighbors import KNeighborsClassifier



X = main_u.inverse_sc(main_u.X.to_numpy(), np.loadtxt('/home/silver/PycharmProjects/AAEDRL/AAE/99.txt'))
X_train, X_test, y_train, y_test = main_u.vertical_split(X, main_u.df["attack_cat"])
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

def xgb_class():
    best_params = {'booster': 'dart', 'lambda': 3.8155711333711397e-07, 'alpha': 4.65598758613365e-15,
               'subsample': 0.9995193751264522, 'colsample_bytree': 0.8316346076096441, 'max_depth': 24,
               'min_child_weight': 5, 'eta': 0.12275341626826634, 'gamma': 2.4961000956964835e-10,
               'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'tree',
               'rate_drop': 3.778486807843725e-05, 'skip_drop': 2.9109736020737014e-07, "verbosity": 0,
               "objective": "multi:softmax", "num_class": 22}

    clf = XGBClassifier(**best_params)
    test_clf = xgb.train(best_params, dtrain)
    return clf, test_clf

def KNN_class():
    clf = KNeighborsClassifier(n_neighbors=16, metric="manhattan", leaf_size=16)
    clf.fit(X_train, y_train)
    return clf


"""_______________________________________________________train______________________________________________________"""
clf, test_clf = xgb_class()
y_pred = cross_val_predict(clf, X_train, y_train, cv=5, method="predict_proba")
report = classification_report(y_train, y_pred)


"""_______________________________________________________test_______________________________________________________"""

y_pred = test_clf.predict(dvalid)
pred_labels = np.rint(y_pred)
roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
# report = classification_report(y_test, y_pred)


# GB [I 2024-12-11 00:14:15,485] Trial 0 finished with value: 0.753581948152614 and parameters: {'gb_n_estimators': 28, 'gb_learning_rate': 0.04836473659876455, 'gb_max_depth': 11}. Best is trial 0 with value: 0.753581948152614.

# RF [I 2024-12-10 13:28:03,055] Trial 29 finished with value: 0.7544742054947895 and parameters: {'rf_max_depth': 16, 'rf_n_estimators': 134}. Best is trial 29 with value: 0.7544742054947895.
