import numpy as np

import xgboost as xgb
from sklearn.metrics import classification_report

from data import main_u
from clfs import clf_optim


dtrain = xgb.DMatrix(clf_optim.X_train, label=clf_optim.y_train)
dvalid = xgb.DMatrix(clf_optim.X_test, label=clf_optim.y_test)
#
best_params = {"verbosity": 0,
        "objective": "multi:softmax",
        "num_class": 14, 'booster': 'dart', 'lambda': 0.00011829588548865235, 'alpha': 7.585211410384668e-08, 'subsample': 0.969570310489219, 'colsample_bytree': 0.7736662472449969, 'max_depth': 46, 'min_child_weight': 3, 'eta': 0.47353592068250144, 'gamma': 0.01228933851098253, 'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'forest', 'rate_drop': 8.53305012365533e-07, 'skip_drop': 0.39742497472410715}
bst = xgb.train(best_params, dtrain)
preds = bst.predict(dvalid)
pred_labels = np.rint(preds)
# recall = sklearn.metrics.recall_score(y_test, pred_labels, average = "weighted")
# print(recall)
class_report = classification_report(clf_optim.y_test, pred_labels)
print(class_report)

# [I 2024-12-10 01:04:00,512] Trial 121 finished with value: 0.7564524797089667 and parameters: {'booster': 'gbtree', 'lambda': 0.22684520981704684, 'alpha': 2.177248101141601e-11, 'subsample': 0.8042169291757324, 'colsample_bytree': 0.7564349666652507, 'max_depth': 17, 'min_child_weight': 3, 'eta': 0.1265683005603432, 'gamma': 5.7751434224941624e-05, 'grow_policy': 'lossguide'}. Best is trial 121 with value: 0.7564524797089667.
