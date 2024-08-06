import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold, GridSearchCV, train_test_split

import main
import new_dataset

df = new_dataset.df

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
grid_search.fit(df)
best_params = grid_search.best_params_

iso = IsolationForest(n_estimators= 50, max_samples = 'auto', contamination = 0.01, max_features = 1.0,
                      bootstrap = True, random_state = 45, warm_start = False).fit(df)

pred = iso.predict(main.y_test)


