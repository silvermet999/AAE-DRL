import numpy as np

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_predict
from data import main_u

X = main_u.inverse_sc(main_u.X.to_numpy(), np.loadtxt('/home/silver/PycharmProjects/AAEDRL/AAE/99.txt'))
X_train, X_test, y_train, y_test = main_u.vertical_split(X, main_u.df["attack_cat"])
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)
#
best_params = {'booster': 'dart', 'lambda': 3.8155711333711397e-07, 'alpha': 4.65598758613365e-15,
               'subsample': 0.9995193751264522, 'colsample_bytree': 0.8316346076096441, 'max_depth': 24,
               'min_child_weight': 5, 'eta': 0.12275341626826634, 'gamma': 2.4961000956964835e-10,
               'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'tree',
               'rate_drop': 3.778486807843725e-05, 'skip_drop': 2.9109736020737014e-07, "verbosity": 0,
               "objective": "multi:softmax", "num_class": 22}


clf = XGBClassifier(**best_params)
bst = xgb.train(best_params, dtrain)
preds = bst.predict(dvalid)
pred_labels = np.rint(preds)
# y_pred = cross_val_predict(clf, X_train, y_train, cv=5, method='predict_proba')
#
class_report = classification_report(y_test, pred_labels)
print(class_report)

# y_train_binarized = label_binarize(y_train, classes=np.arange(7))
# preds_binarized = label_binarize(y_pred, classes=np.arange(7))
# fpr = {}
# tpr = {}
# roc_auc = {}
# for i in range(7):
#     fpr[i], tpr[i], _ = roc_curve(y_train_binarized[:, i], preds_binarized[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
# average_roc_auc = np.mean(list(roc_auc.values()))
# print(f"Average ROC AUC: {average_roc_auc}")

# roc_auc = roc_auc_score(y_train, y_pred, multi_class='ovr')
