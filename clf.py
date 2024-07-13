import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import KFold, GridSearchCV, StratifiedShuffleSplit, train_test_split, cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
from matplotlib.legend_handler import HandlerLine2D
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import xgboost
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, ADASYN
import main


synth = pd.read_csv("runs/100009.csv")
synth2 = pd.read_csv("runs/24.csv")
synth3 = pd.read_csv("runs/1000011.csv")
df_synth1 = pd.DataFrame(synth)
df_synth1 = df_synth1.rename(columns = dict(zip(df_synth1.columns, main.X.columns)))
df_synth2 = pd.DataFrame(synth2)
df_synth2 = df_synth2.rename(columns = dict(zip(df_synth2.columns, main.X.columns)))
df_synth3 = pd.DataFrame(synth3)
df_synth3 = df_synth3.rename(columns = dict(zip(df_synth3.columns, main.X.columns)))
X = pd.DataFrame(main.X_rs)
X = X.rename(columns = dict(zip(X.columns, main.X.columns)))
frames_un = [df_synth1, df_synth2, X]
df_synth = pd.concat(frames_un)

synth_sup = pd.read_csv("runs/17.csv")
df_synth_sup = pd.DataFrame(synth_sup)
df_synth_sup = df_synth_sup.rename(columns = dict(zip(df_synth_sup.columns, main.df_enc.columns)))
df_enc = pd.DataFrame(main.df_enc)
df_enc = df_enc.rename(columns = dict(zip(df_enc.columns, main.df_enc.columns)))
frames_sup = [df_synth_sup, df_enc]
df_synths = pd.concat(frames_sup)




def xgb():
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
        num_class=14,
    )

    # search_spaces = {
    #     'n_estimators': space.Integer(93, 181),
    #     'max_depth': space.Integer(9, 17),
    #     'learning_rate': space.Real(0.01, 0.24, 'log-uniform')
    # }


    # opt = BayesSearchCV(
    #     clf,
    #     search_spaces=search_spaces,
    #     n_iter=10,
    #     random_state=0,
    #     verbose=1,
    #     n_jobs=-1,
    #     scoring='accuracy',
    #     cv=3
    # )

    le = LabelEncoder()
    y = le.fit_transform(main.y["Category"])
    sample_weights = np.ones_like(y, dtype=float)
    sample_weights[3] = 2
    sample_weights[9] = 2.0
    sample_weights[13] = 3.0

    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    sm = SMOTE(random_state=42)
    scores = []

    for train_index, val_index in kf.split(main.df):
        X_tr, X_val = df_synths.iloc[train_index], df_synths.iloc[val_index]
        y_tr, y_val = y[train_index], y[val_index]
        sample_weights_tr = sample_weights[train_index]

        X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
        original_sample_count = len(y_tr)
        synthetic_sample_count = len(y_tr_res) - original_sample_count
        synthetic_sample_weight = np.mean(sample_weights_tr)
        sample_weights_res = np.concatenate([
            sample_weights_tr,
            np.full(synthetic_sample_count, synthetic_sample_weight)
        ])

        clf.fit(X_tr_res, y_tr_res, sample_weight=sample_weights_res)
        y_pred = clf.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred))

        class_report = classification_report(y_val, y_pred)

        with open('xgb/xgb_kf10_smote_synth_sup_classification_report.txt', 'w') as f:
            f.write(class_report)

    return y_pred, class_report


def objective(params):
    model = xgboost.XGBClassifier(
        objective='multi:softmax',
        seed=0,
        **params
    )

    # Perform cross-validation
    score = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy').mean()

    # We want to maximize accuracy, but Hyperopt minimizes the objective function
    return {'loss': -score, 'status': STATUS_OK}

# Define the search space
space_xgb = {
    'max_depth': hp.choice('max_depth', [14, 16, 18]),
    'gamma': hp.uniform('gamma', 0.01, 0.018),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.03),
    'reg_alpha': hp.uniform('reg_alpha', 2, 4),
    'reg_lambda': hp.uniform('reg_lambda', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 0.68),
    'min_child_weight': hp.choice('min_child_weight', [1, 2, 3]),
    'n_estimators': hp.choice('n_estimators', [160, 180, 200])
}

# Prepare the data
le = LabelEncoder()
ynp = le.fit_transform(main.y["Category"])
y = pd.DataFrame(ynp, columns=['Category'])
x_train, x_test, y_train, y_test = train_test_split(df_synth_sup[:39744], y, test_size=.2)

# Run the optimization
trials = Trials()
best = fmin(fn=objective,
            space=space_xgb,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials,
            verbose=2)




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
    with open('adaboost/adaboost_classification_report.txt', 'w') as f:
        f.write(class_report)
    return class_report



# trials = Trials()
# best = fmin(fn=objective,
#             space=space_ada,
#             algo=tpe.suggest,
#             max_evals=100,
#             trials=trials)
#
# print("Best hyperparameters:", best)




le = LabelEncoder()
y_train_bin = le.fit_transform(main.y_train["Category"])
y_test_bin = le.fit_transform(main.y_test["Category"])

space_gbc = {
    "learning_rate": [0.25, 0.1, 0.05, 0.01],
    "n_estimators": [8, 16, 32, 64, 100, 200],
    "max_depth": list(range(1, 33)),
    "min_samples_split": np.arange(0.1, 1.1, 0.1),
    "min_samples_leaf": np.arange(0.1, 0.6, 0.1),
    "max_features": list(range(1, main.x_train.shape[1] + 1))
}


def gbc():
    clf = GradientBoostingClassifier(
        max_depth=16,
        n_estimators=100,
        learning_rate=.01,
        min_samples_leaf=10,
        max_features="sqrt",
        min_samples_split=10,
    )
    le = LabelEncoder()
    y = le.fit_transform(main.y["Category"])
    sample_weights = np.ones_like(y, dtype=float)
    sample_weights[3] = 2.0
    sample_weights[9] = 2.0
    sample_weights[13] = 3.0

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(main.x_train):
        X_tr, X_val = main.x_train[train_index], main.x_train[val_index]
        y_tr, y_val = y[train_index], y[val_index]
        sample_weights_tr = sample_weights[train_index]

        clf.fit(X_tr, y_tr, sample_weight=sample_weights_tr)
        y_pred = clf.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred))
        class_report = classification_report(y_val, y_pred)
        with open('gbc_classification_report.txt', 'w') as f:
            f.write(class_report)
    return class_report


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