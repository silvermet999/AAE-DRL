from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC
from xgboost import XGBClassifier
import dim_reduction
import main


ad = AdaBoostClassifier()
gbc = GradientBoostingClassifier()
iso = IsolationForest()
svm = SVC()
xgbc = XGBClassifier()



def decision_tree():
    dtc = tree.DecisionTreeClassifier()
    dtc = dtc.fit(dim_reduction.x_pca_train, main.y_train)
    pred = dtc.predict(dim_reduction.x_pca_test)
    return dtc, pred

