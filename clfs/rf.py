import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import label_binarize

from data import main_u

# [I 2024-12-10 13:28:03,055] Trial 29 finished with value: 0.7544742054947895 and parameters: {'rf_max_depth': 16, 'rf_n_estimators': 134}. Best is trial 29 with value: 0.7544742054947895.


X = main_u.inverse_sc(main_u.X.to_numpy(), np.loadtxt('/home/silver/PycharmProjects/AAEDRL/AAE/smp2.txt'))
X_train, X_test, y_train, y_test = main_u.vertical_split(X, main_u.df["attack_cat"])
clf = RandomForestClassifier(rf_max_depth= 16, rf_n_estimators= 134)

y_pred = cross_val_predict(clf, X_train, y_train, cv=5)
report_t = classification_report(y_train, y_pred)
print(report_t)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# y_test_binarized = label_binarize(y_test, classes=np.arange(14))
# preds_binarized = label_binarize(y_pred, classes=np.arange(14))
# fpr = {}
# tpr = {}
# roc_auc = {}
# for i in range(14):
#     fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], preds_binarized[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
# average_roc_auc = np.mean(list(roc_auc.values()))
# print(f"Average ROC AUC: {average_roc_auc}")
