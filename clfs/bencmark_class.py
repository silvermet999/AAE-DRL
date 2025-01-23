import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_predict
from data import main_u
from sklearn.neighbors import KNeighborsClassifier



df = pd.DataFrame(pd.read_csv("/home/silver/PycharmProjects/AAEDRL/AAE/ds2.csv"))[:141649]
df_disc, df_cont = main_u.df_type_split(df)
_, mainX_cont = main_u.df_type_split(main_u.X)
X_inv = main_u.inverse_sc_cont(mainX_cont, df_cont)
X = df_disc.join(X_inv)

X_train, X_test, y_train, y_test = main_u.vertical_split(X, main_u.y)
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

def xgb_class():
    best_params = {'booster': 'dart', 'lambda': 0.002777166262623694, 'alpha': 0.19076297470570558,
                   'subsample': 0.8249767279788564, 'colsample_bytree': 0.9752618433258462, 'max_depth': 38,
                   'min_child_weight': 3, 'eta': 0.06420221699462178, 'gamma': 0.022903757823417577,
                   'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'forest',
                   'rate_drop': 0.00020154770959263124, 'skip_drop': 2.858519185587543e-07, "verbosity": 0,
                   "objective": "multi:softmax", "num_class": 30}

    clf = XGBClassifier(**best_params)
    clf.fit(X_train, y_train)
    # y_pred = cross_val_predict(clf, X_train, y_train, cv=5, method="predict_proba")
    # y_pred_max = np.argmax(y_pred, axis=1)
    # report_train = classification_report(y_train, y_pred_max)
    y_proba = clf.predict_proba(X_test)
    lb = LabelBinarizer()
    y_test_binarized = lb.fit_transform(y_test)
    auc_scores = []
    for i in range(y_proba.shape[1]):
        auc = roc_auc_score(y_test_binarized[:, i], y_proba[:, i])
        auc_scores.append(auc)
        print(f"AUC-ROC for class {i}: {auc:.4f}")
    macro_auc = roc_auc_score(y_test_binarized, y_proba, average="macro")
    print(f"Macro-Average AUC-ROC: {macro_auc:.4f}")


    test_clf = xgb.train(best_params, dtrain)
    y_pred_val = test_clf.predict(dvalid)
    report_test = classification_report(y_test, y_pred_val)
    return macro_auc, report_test

def KNN_class():
    # on smp2
    clf = KNeighborsClassifier(n_neighbors= 11, metric = 'euclidean', leaf_size = 72)
    clf.fit(X_train, y_train)
    y_pred = cross_val_predict(clf, X_train, y_train, cv=5, method="predict_proba")
    y_pred_max = np.argmax(y_pred, axis=1)
    report_train = classification_report(y_train, y_pred_max)
    roc_auc_train = roc_auc_score(y_train, y_pred, multi_class='ovr')
    y_pred_val = clf.predict(X_test)
    report_test = classification_report(y_test, y_pred_val)
    lb = LabelBinarizer()
    y_proba = clf.predict_proba(X_test)
    y_test_binarized = lb.fit_transform(y_test)
    auc_scores = []
    for i in range(y_proba.shape[1]):
        auc = roc_auc_score(y_test_binarized[:, i], y_proba[:, i])
        auc_scores.append(auc)
        print(f"AUC-ROC for class {i}: {auc:.4f}")
    macro_auc = roc_auc_score(y_test_binarized, y_proba, average="macro")
    print(f"Macro-Average AUC-ROC: {macro_auc:.4f}")
    return report_test, macro_auc

def rf_class():
    # on smp2
    clf = RandomForestClassifier(
        max_depth = 14, n_estimators = 181
    )
    clf.fit(X_train, y_train)
    y_pred = cross_val_predict(clf, X_train, y_train, cv=5, method="predict_proba")
    y_pred_max = np.argmax(y_pred, axis=1)
    report_train = classification_report(y_train, y_pred_max)
    roc_auc_train = roc_auc_score(y_train, y_pred, multi_class='ovr')
    y_pred_val = clf.predict(X_test)
    report_test = classification_report(y_test, y_pred_val)
    lb = LabelBinarizer()
    y_proba = clf.predict_proba(X_test)
    y_test_binarized = lb.fit_transform(y_test)
    auc_scores = []
    for i in range(y_proba.shape[1]):
        auc = roc_auc_score(y_test_binarized[:, i], y_proba[:, i])
        auc_scores.append(auc)
        print(f"AUC-ROC for class {i}: {auc:.4f}")
    macro_auc = roc_auc_score(y_test_binarized, y_proba, average="macro")
    print(f"Macro-Average AUC-ROC: {macro_auc:.4f}")
    return report_test, macro_auc

def gb_class():
    clf = GradientBoostingClassifier(
        n_estimators= 19, learning_rate = 0.07232257422800473, max_depth = 10
    )
    clf.fit(X_train, y_train)
    y_pred = cross_val_predict(clf, X_train, y_train, cv=5, method="predict_proba")
    y_pred_max = np.argmax(y_pred, axis=1)
    report_train = classification_report(y_train, y_pred_max)
    roc_auc_train = roc_auc_score(y_train, y_pred, multi_class='ovr')
    y_pred_val = clf.predict(X_test)
    report_test = classification_report(y_test, y_pred_val)

    lb = LabelBinarizer()
    y_proba = clf.predict_proba(X_test)
    y_test_binarized = lb.fit_transform(y_test)
    auc_scores = []
    for i in range(y_proba.shape[1]):
        auc = roc_auc_score(y_test_binarized[:, i], y_proba[:, i])
        auc_scores.append(auc)
        print(f"AUC-ROC for class {i}: {auc:.4f}")
    macro_auc = roc_auc_score(y_test_binarized, y_proba, average="macro")
    print(f"Macro-Average AUC-ROC: {macro_auc:.4f}")
    return report_test, macro_auc
