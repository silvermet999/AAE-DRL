import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

from AAE import main

X_train = main.X_train_rs
y_train = main.y_train
X_test = main.X_test_rs
y_test = main.y_test

X_train, y_train = SMOTE().fit_resample(X_train, y_train)
clf = KNeighborsClassifier(10, leaf_size=36, p=1, metric="minkowski")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
report = classification_report(y_train, y_pred)
print(report)
y_train_binarized = label_binarize(y_train, classes=np.arange(14))
preds_binarized = label_binarize(y_pred, classes=np.arange(14))
fpr = {}
tpr = {}
roc_auc = {}
for i in range(14):
    fpr[i], tpr[i], _ = roc_curve(y_train_binarized[:, i], preds_binarized[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
average_roc_auc = np.mean(list(roc_auc.values()))
print(f"Average ROC AUC: {average_roc_auc}")

# [I 2024-09-05 02:39:46,705] Trial 11 finished with value: 0.9671570154871914 and parameters: {'n_neighbors': 1, 'metric': 'minkowski', 'p': 1, 'leaf_size': 20}. Best is trial 11 with value: 0.9671570154871914.
