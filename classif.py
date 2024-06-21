import numpy as np
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC
from xgboost import XGBClassifier
import dim_reduction


ad = AdaBoostClassifier()
gbc = GradientBoostingClassifier()
iso = IsolationForest()
svm = SVC()
xgbc = XGBClassifier()



def classifier(real_data, gen_data):
    X = np.vstack((real_data, gen_data))
    y = np.hstack((np.ones(real_data.shape[0]), np.zeros(gen_data.shape[0])))
    dtc = tree.DecisionTreeClassifier()
    dtc = dtc.fit(X, y)
    pred = dtc.predict(dim_reduction.x_pca_test)
    return dtc, pred

