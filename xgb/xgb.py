import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, ADASYN
import main
import torch

cuda = True if torch.cuda.is_available() else False
torch_gpu = torch.empty((15000, 15000)).cuda()
torch.cuda.memory_allocated()


synth = pd.read_csv("runs/100009.csv")
df_synth = pd.DataFrame(synth)
df_synth = df_synth.rename(columns = dict(zip(df_synth.columns, main.X.columns)))
X = pd.DataFrame(main.X_rs)
X = X.rename(columns = dict(zip(X.columns, main.X.columns)))
frames_un = [df_synth, X]
df_synth = pd.concat(frames_un)





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
        X_tr, X_val = df_synth.iloc[train_index], df_synth.iloc[val_index]
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

        with open('xgb_kf10_smote_synth_sup_classification_report.txt', 'w') as f:
            f.write(class_report)

    return y_pred, class_report


space = {
    'max_depth': hp.quniform('max_depth', 14, 30, 1),
    'gamma': hp.uniform('gamma', 0.1, 1),
    'learning_rate': hp.uniform('learning_rate', np.log(0.01), np.log(0.1)),
    'reg_alpha': hp.quniform('reg_alpha', 90, 200, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.4, 0.9),
    'min_child_weight': hp.quniform('min_child_weight', 10, 40, 1),
    'max_delta_step' : hp.uniform('max_delta_step', 0, 5)
}
def objective(space):
    clf=xgboost.XGBClassifier(
                    n_estimators =180, max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']), learning_rate=space['learning_rate'],
                                         device='cuda')

    le = LabelEncoder()
    y = le.fit_transform(main.y["Category"])

    X_train, X_test, y_train, y_test = train_test_split(df_synth[:39744], y, test_size=.2)

    evaluation = [(X_train, y_train), (X_test, y_test)]

    clf.fit(X_train, y_train,
            eval_set=evaluation)

    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    return {'loss': -accuracy, 'status': STATUS_OK }
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
            verbose=2)
print("Best hyperparameters:", best)
