import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import dim_reduction
import main
from scipy.stats import mode




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
    clf = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='multi:softmax', n_class = 14)
    clf.fit(main.x_train, y_train)
    pred = clf.predict(main.x_test)
    class_report = classification_report(y_test[:10688], pred)
    with open('xgb_classification_report.txt', 'w') as f:
        f.write(class_report)
    return pred, class_report