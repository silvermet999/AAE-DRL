import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import recall_score

from sklearn.svm import SVC
import xgboost
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.utils import compute_sample_weight

import dim_reduction
import main
from scipy.stats import mode
from skopt import BayesSearchCV, space




def classifier():
    dtc = tree.DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=2, min_samples_leaf=2)
    dtc = dtc.fit(main.x_train, main.y_train)
    pred = dtc.predict(main.x_test)
    class_report = classification_report(main.y_test.iloc[:10688, 0], pred[:, 0])
    graph = tree.plot_tree(dtc, feature_names=main.X.columns, class_names=main.y.columns, filled=True, rounded=True)
    plt.savefig('tree.png')
    with open('classification_report.txt', 'w') as f:
        f.write(class_report)
    return pred, graph, class_report



def adaboost():
    base_estimator = tree.DecisionTreeClassifier(max_depth=1)
    clf = AdaBoostClassifier(base_estimator, n_estimators=100, algorithm='SAMME', random_state=42)
    clf = clf.fit(main.x_train, main.y_train.iloc[:, 0])
    pred = clf.predict(main.x_test)
    class_report = classification_report(main.y_test.iloc[:10688, 0], pred)
    with open('ada_classification_report.txt', 'w') as f:
        f.write(class_report)

    return pred, class_report



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

def xgb():
    clf = xgboost.XGBClassifier(n_estimators=14, max_depth=5, learning_rate=0.1, objective='multi:softmax', n_class = 14)
    clf.fit(main.x_train, y_train)
    pred = clf.predict(main.x_test)
    label_mapping = {i: label for i, label in enumerate(np.unique(y_train))}
    class_report = classification_report(y_test[:10688], pred, target_names=[label_mapping[i] for i in np.unique(y_train)])
    with open('xgb_classification_report.txt', 'w') as f:
        f.write(class_report)
    return pred, class_report


sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(main.x_train, y_train)

dtrain = xgboost.DMatrix(X_res, label=y_res)
dtest = xgboost.DMatrix(main.x_test, label=y_test)



def xgb_bscv():
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_res, y_res = smote.fit_resample(main.x_train, y_train)
    clf = xgboost.XGBClassifier(n_estimators=20, max_depth=10, gamma=0.1, reg_alpha=1.0,
                            reg_lambda=1.0, learning_rate=0.1, objective='multi:softmax',
                            num_class=14)

    opt = BayesSearchCV(
        clf,
        search_spaces={
            'n_estimators': space.Integer(10, 100),
            'max_depth': space.Integer(5, 10),
            'learning_rate': space.Real(0.01, 1.0, 'log-uniform')
        },
        n_iter=10,
        random_state=0,
        verbose=1,
        n_jobs=-1,
        scoring='accuracy',
        cv=3
    )

    pipeline = Pipeline([('smote', smote), ('opt', opt)])

    pipeline.fit(X_res, y_res)
    pred = pipeline.predict(main.x_test)
    class_report = classification_report(y_test[:10688], pred)

    with open('xgb_bscv_smote_tweek_classification_report.txt', 'w') as f:
        f.write(class_report)

    return pred, class_report


space = {
    'max_depth': hp.quniform("max_depth", 3, 18, 1),
    'gamma': hp.uniform('gamma', 0, 10),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'reg_alpha': hp.quniform('reg_alpha', 0, 180, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'n_estimators': 180,
    'seed': 0
}

# Compute class weights
class_weights = compute_sample_weight('balanced', y_train)


# Define the objective function
def objective():
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
        num_class=14
    )

    clf.fit(main.x_train, y_train, sample_weight=class_weights)
    pred = clf.predict(main.x_test)
    recall = recall_score(y_test[:10688], pred, average='macro')

    return recall

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)


# Train the final model with the best hyperparameters
best_model = xgboost.XGBClassifier(
    max_depth=int(best['max_depth']),
    gamma=best['gamma'],
    learning_rate=best['learning_rate'],
    reg_alpha=int(best['reg_alpha']),
    reg_lambda=best['reg_lambda'],
    colsample_bytree=best['colsample_bytree'],
    min_child_weight=int(best['min_child_weight']),
    n_estimators=best['n_estimators'],
    seed=best['seed'],
    objective='multi:softmax',
    num_class=14
)

best_model.fit(main.x_train, y_train, sample_weight=class_weights)
pred = best_model.predict(main.x_test)
class_report = classification_report(y_test[:10688], pred)

with open('xgb_hyperopt_classification_report.txt', 'w') as f:
    f.write(class_report)