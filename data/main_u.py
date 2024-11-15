"""-----------------------------------------------import libraries-----------------------------------------------"""
import os
from collections import defaultdict

from imblearn.pipeline import make_pipeline
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler, MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from fitter import Fitter, get_common_distributions
from ydata_profiling import ProfileReport
import warnings
warnings.filterwarnings('ignore')





"""--------------------------------------------data exploration/cleaning--------------------------------------------"""
train = pd.read_csv('/home/silver/UNSW_NB15_training-set.csv')
test = pd.read_csv('/home/silver/UNSW_NB15_testing-set.csv')

dfs = [train, test]
df = pd.concat(dfs, ignore_index=True)
df = df.drop(df.columns[df.nunique() == 1], axis = 1) # no change
df = df.drop(df.columns[df.nunique() == len(df)], axis = 1) # no change


df["proto"].replace("a/n", np.nan, inplace=True)
df["service"].replace("-", np.nan, inplace=True)
df["state"].replace("no", np.nan, inplace=True)

df.fillna('Missing', inplace=True)


le = LabelEncoder()
cols_le = ["attack_cat", "proto", "service", "state"]

for i in cols_le:
    df[i] = le.fit_transform(df[i])
#     label_mappings[i] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
# for col, mapping in label_mappings.items():
#     print(f"Mapping for {col}: {mapping}")

for col in cols_le:
    value_counts = df[col].value_counts()
    singletons = value_counts[value_counts == 1].index
    df = df[~df[col].isin(singletons)] # 34549, 129 => deleted 12483 samples

def cap_outliers_iqr(df, factor=40):
    capped_df = df.copy()
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_threshold = Q1 - factor * IQR
        upper_threshold = Q3 + factor * IQR
        capped_df[column] = np.where(df[column] < lower_threshold, lower_threshold, df[column])
        capped_df[column] = np.where(capped_df[column] > upper_threshold, upper_threshold, capped_df[column])
    return capped_df
# df = cap_outliers_iqr(df)


def corr(df):
    correlation = df.corr()
    f_corr = {}
    for column in correlation.columns:
        correlated_with = list(correlation.index[(correlation[column] >= 0.75) | (correlation[column] <= -0.75)])
        for corr_col in correlated_with:
            if corr_col != column:
                df_corr = correlation.loc[corr_col, column]
                f_corr[(column, corr_col)] = df_corr
    f_corr = pd.DataFrame.from_dict(f_corr, orient="index")
    f_corr = f_corr.drop_duplicates()
    # f_corr.to_csv("f_corr.csv")
    columns_to_drop = {col_pair[1] for col_pair in f_corr.index}
    df = df.drop(columns=columns_to_drop)
    return df


"""-----------------------------------------------vertical data split-----------------------------------------------"""
X = corr(df.drop(["attack_cat", "label"], axis = 1))
y = df["attack_cat"]

y = pd.get_dummies(y).astype(int)
cols = ['Normal', 'Backdoor', 'Analysis', 'Fuzzers', 'Shellcode',
       'Reconnaissance', 'Exploits', 'DoS', 'Worms', 'Generic']
y = y.rename(columns = dict(zip(y.columns, cols)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

def min_max(df):
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    return df

X_train_sc = min_max(X_train)
X_test_sc = min_max(X_test)
y_train = y_train.to_numpy()



# profile.to_file("report_u.html")
#
#
# def kolmogorov_smirnov_test(column, dist='norm'):
#     D, p_value = stats.kstest(column, dist)
#     return p_value > 0.05
#
# def anderson_darling_test(column, dist='norm'):
#     result = stats.anderson(column, dist=dist)
#     return result.statistic < result.critical_values[2]
#
# def fit_distributions(column):
#     f = Fitter(column, distributions=get_common_distributions())
#     f.fit()
#     f.summary()
#     best_fit = f.get_best(method='sumsquare_error')
#     # best_fit_distribution = list(best_fit.keys())[0]
#
#     return best_fit
#     return best_fit_distribution
#
# X_train_sc = pd.DataFrame(X_train_sc)
# results = {}
#
# for idx, column in enumerate(X_train_sc.columns):
#     col_results = []
#     if kolmogorov_smirnov_test(X_train_sc[column]):
#         col_results.append(f"Column {column} follows the specified distribution (Kolmogorov-Smirnov test).")
#     if anderson_darling_test(X_train_sc[column]):
#         col_results.append(f"Column {column} follows the specified distribution (Anderson-Darling test).")
#     best_fit = fit_distributions(X_train_sc[column])
#     col_results.append(f"{best_fit}")
#     results[idx] = col_results
#
# for idx in range(28):
#     if idx in results:
#         print(f"{idx}")
#         for result in results[idx]:
#             print(result)
#         print()
#     else:
#         print(f"{idx}: No data\n")

