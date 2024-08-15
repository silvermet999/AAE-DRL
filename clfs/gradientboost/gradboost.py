import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, ADASYN
from AAE import main


# [I 2024-08-13 01:27:39,309] Trial 33 finished with value: 0.9802483588820078 and parameters: {'classifier': 'gb', 'gb_n_estimators': 200, 'gb_learning_rate': 0.049662646712519194, 'gb_max_depth': 9}. Best is trial 33 with value: 0.9802483588820078.

X_train = main.X_train_rs
y_train = main.y_train
X_test = main.X_test_rs
y_test = main.y_test

X_train, y_train = SMOTE().fit_resample(X_train, y_train)


clf = GradientBoostingClassifier(
    max_depth=9,
    n_estimators=200,
    learning_rate=.049662646712519194,
    # subsample=1.0,
    # criterion='friedman_mse',
    # min_samples_split=2,
    # min_samples_leaf=1,
    # min_weight_fraction_leaf=0.0,
    # min_impurity_decrease=0.0,
    # init=None,
    # random_state=None,
    # max_features=None,
    # verbose=0,
    # max_leaf_nodes=None,
    # warm_start=False

)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
class_report = classification_report(y_test, pred)
print(class_report)
