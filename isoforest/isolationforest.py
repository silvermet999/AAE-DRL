import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, ADASYN
import main




def isolationforest():
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples': ['auto', 0.1, 0.5, 1.0],
        'contamination': [0.01, 0.05, 0.1, 0.2],
        'max_features': [0.5, 0.8, 1.0],
        'bootstrap': [True, False],
        'random_state' : [0, 24, 45],
        'warm_start' : [True, False],

    }

    iso_forest = IsolationForest(n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=iso_forest,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(main.X_rs)

    best_params = grid_search.best_params_
    return best_params