import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

from data import main_u

X = main_u.inverse_sc(main_u.X.to_numpy(), np.loadtxt('/home/silver/PycharmProjects/AAEDRL/AAE/smp2.txt'))
X_train, X_test, y_train, y_test = main_u.vertical_split(X, main_u.df["attack_cat"])


clf = KNeighborsClassifier(n_neighbors= 16, metric = "manhattan", leaf_size= 16)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
# y_train_binarized = label_binarize(y_train, classes=np.arange(14))
# preds_binarized = label_binarize(y_pred, classes=np.arange(14))
# fpr = {}
# tpr = {}
# roc_auc = {}
# for i in range(14):
#     fpr[i], tpr[i], _ = roc_curve(y_train_binarized[:, i], preds_binarized[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
# average_roc_auc = np.mean(list(roc_auc.values()))
# print(f"Average ROC AUC: {average_roc_auc}")

# [I 2024-12-10 03:00:17,444] Trial 9 finished with value: 0.749947463612092 and parameters: {'n_neighbors': 16, 'metric': 'manhattan', 'leaf_size': 16}. Best is trial 9 with value: 0.749947463612092.
