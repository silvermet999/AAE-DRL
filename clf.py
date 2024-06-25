import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import dim_reduction
import main
from scipy.stats import mode


ad = AdaBoostClassifier()
gbc = GradientBoostingClassifier()
iso = IsolationForest()
svm = SVC()
xgbc = XGBClassifier()


def classifier():
    dtc = tree.DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=2, min_samples_leaf=2)
    dtc = dtc.fit(main.x_train, main.y_train)
    pred = dtc.predict(main.x_test)
    class_report = classification_report(main.y_test.iloc[:10688, 0], pred[:, 0])
    graph = tree.plot_tree(dtc, feature_names=main.X.columns, class_names=main.y.columns, filled=True, rounded=True)
    plt.savefig('tree.png')
    with open('classification_report.txt', 'w') as f:
        f.write(class_report)
    return pred, class_report

clf_pred, class_report = classifier()
print(clf_pred)
