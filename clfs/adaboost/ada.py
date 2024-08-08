import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, ADASYN
import main
import torch

cuda = True if torch.cuda.is_available() else False
torch_gpu = torch.empty((15000, 15000)).cuda()
torch.cuda.memory_allocated()
def adaboost():
    space_ada = {
        'n_estimators': hp.choice('n_estimators', range(50, 500)),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
        'algorithm': hp.choice('algorithm', ['SAMME'])
    }
    clf = AdaBoostClassifier(
        n_estimators=400,
        learning_rate=1
    )
    # grid_search = GridSearchCV(
    #     estimator=clf,
    #     param_grid=param_grid,
    #     cv=5,
    #     n_jobs=-1,
    #     verbose=2,
    #     scoring='accuracy'
    # )
    # grid_search.fit(main.x_train, main.y_train["Category"])
    # best_model = grid_search.best_estimator_
    le = LabelEncoder()
    y_train = le.fit_transform(main.y_train["Category"])
    y_test = le.fit_transform(main.y_test["Category"])
    clf.fit(main.x_train, y_train)
    y_pred = clf.predict(main.x_test)
    class_report = classification_report(y_test, y_pred)
    with open('adaboost_classification_report.txt', 'w') as f:
        f.write(class_report)
    return class_report