import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MaxAbsScaler

from data import main
import pandas as pd

X = main.X
y = main.y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split(X_cl, y_cl, test_size=0.2, random_state=42)


def robust_scaler(df):
    scaler = RobustScaler()
    df = scaler.fit_transform(df)
    return df

X_train_rs = robust_scaler(X_train)
# X_train_rs_cl = robust_scaler(X_train_cl)

X_test_rs = robust_scaler(X_test)
# X_test_rs_cl = robust_scaler(X_test_cl)

# # np.isnan(X_rs).any()


def max_abs_scaler(df):
    scaler = MaxAbsScaler()
    df = scaler.fit_transform(df)
    return df

def _inverse_transform_continuous():
    scaled_data = RobustScaler().fit_transform(X_train)
    data = pd.DataFrame(X).astype(float)
    selected_normalized_value = np.random.normal(data.iloc[:, 0], 0.1)
    data.iloc[:, 0] = selected_normalized_value

    return scaled_data.reverse_transform(data)

# X_train_mas = max_abs_scaler(X_train)
# X_train_mas_cl = max_abs_scaler(X_train_cl)
#
# X_test_mas = max_abs_scaler(X_test)
# X_test_mas_cl = max_abs_scaler(X_test_cl)