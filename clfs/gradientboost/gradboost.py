import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, ADASYN
import main




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
        with open('../gbc_classification_report.txt', 'w') as f:
            f.write(class_report)
    return class_report