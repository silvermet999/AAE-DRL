"""-----------------------------------------------import libraries-----------------------------------------------"""
from collections import defaultdict
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from fitter import Fitter, get_common_distributions
import warnings
warnings.filterwarnings('ignore')





"""--------------------------------------------data exploration/cleaning--------------------------------------------"""
train = pd.read_csv('/home/silver/UNSW_NB15_training-set.csv')
test = pd.read_csv('/home/silver/UNSW_NB15_testing-set.csv')
extra = pd.read_csv("/home/silver/PycharmProjects/AAEDRL/data/dataset.csv")
extra = extra.rename(columns={
    'sintpkt': 'sinpkt',
    'dintpkt': 'dinpkt',
    'ct_src_ ltm': "ct_src_ltm",
    'Label': "label"
})

dfs = [train, test, extra]
df = pd.concat(dfs, ignore_index=True)
df = df.drop(df.columns[df.nunique() == 1], axis = 1) # no change
df = df.drop(df.columns[df.nunique() == len(df)], axis = 1) # no change
df = df.drop(["id", "Unnamed: 0"], axis=1)

df["rate"] = df["rate"].fillna(df["rate"].mean())
df["proto"].replace("a/n", np.nan, inplace=True)
df["service"].replace("-", np.nan, inplace=True)
df["state"].replace("no", np.nan, inplace=True)

df.fillna('Missing', inplace=True)
df["attack_cat"] = df["attack_cat"].replace([' Fuzzers', ' Fuzzers '], "Fuzzers")
df["attack_cat"] = df["attack_cat"].replace("Backdoors", "Backdoor")
df["attack_cat"] = df["attack_cat"].replace(" Reconnaissance ", "Reconnaissance")
df["attack_cat"] = df["attack_cat"].replace(" Shellcode ", "Shellcode")

le = LabelEncoder()
cols_le = ["attack_cat", "proto", "service", "state"]
mappings = {}

for col in cols_le:
    df[col] = le.fit_transform(df[col])
#     mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
# #
# for col, mapping in mappings.items():
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

    # plt.figure(figsize=(30, 20))
    # sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    # plt.title("Correlation Matrix Heatmap")
    # plt.savefig("corr.png")

    # f_corr = pd.DataFrame.from_dict(f_corr, orient="index")
    # f_corr = f_corr.drop_duplicates()
    # columns_to_drop = {col_pair[0] for col_pair in f_corr.index}
    # df = df.drop(columns=columns_to_drop)
    return f_corr

df = df.drop(['ct_srv_src', 'ct_srv_dst', 'ct_src_dport_ltm', 'ct_dst_src_ltm', 'ct_dst_ltm', 'ct_src_ltm',
              'ct_ftp_cmd', 'synack', 'ackdat', 'dloss', 'sbytes', 'stcpb', 'sloss', "dloss", "dbytes", "sbytes",
               'is_sm_ips_ports', 'dwin', 'swin', 'state', 'tcprtt', "is_ftp_login"
               ], axis=1)

generic = df[df['attack_cat'].isin([5])]
genidx = generic.tail(184352).index
df = df.drop(genidx)
# malware_and_low_imp = df[df["attack_cat"].isin([0, 1, 8, 9, 7, 4])]
# exploits = df[df["attack_cat"].isin([3, 2])]
df["attack_cat"] = df["attack_cat"].replace([0, 1, 8, 9, 7, 4], 0)
df["attack_cat"] = df["attack_cat"].replace([[3, 2]], 1)
explidx = df[df['attack_cat'].isin([1])].tail(31756).index
df = df.drop(explidx)


"""-----------------------------------------------vertical data split-----------------------------------------------"""
def vertical_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
    return X_train, X_test, y_train, y_test


X = df.drop(["attack_cat", "label"], axis = 1)
y_bin = df["label"]


X_train, X_test, y_train, y_test = vertical_split(X, y_bin)

def mac(df):
    scaler = MaxAbsScaler()
    df = scaler.fit_transform(df)
    return df


def prep(X):
    X_disc = X[["proto", "service", "ct_state_ttl", "ct_flw_http_mthd", "dttl"]]
    X_cont = mac(X[[feature for feature in X.columns if feature not in X_disc]])
    X_sc = np.concatenate((X_disc, X_cont), axis=1)
    return X_sc


X_train_sc = prep(X_train)
X_test_sc = prep(X_test)

def inverse_sc(X, synth):
    max_abs_values = np.abs(X).max(axis=0)
    synth_inv = synth * max_abs_values
    return synth_inv

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
# for idx in range(22):
#     if idx in results:
#         print(f"{idx}")
#         for result in results[idx]:
#             print(result)
#         print()
#     else:
#         print(f"{idx}: No data\n")

