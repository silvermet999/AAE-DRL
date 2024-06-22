import numpy as np
import pandas as pd
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

df_sel = main.df.iloc[:5000, :100]

def classifier(real_data, gen_data):
    X = np.vstack((real_data, gen_data))
    y = np.hstack((np.ones(real_data.shape[0]), np.zeros(gen_data.shape[0])))
    dtc = tree.DecisionTreeClassifier()
    dtc = dtc.fit(X, y)
    pred = dtc.predict(dim_reduction.x_pca_test)
    # pred_df = pd.DataFrame(pred, columns=df_sel.columns)
    return dtc, pred

