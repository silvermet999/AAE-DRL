"""-----------------------------------------------import libraries-----------------------------------------------"""
import os
from collections import defaultdict

from imblearn.over_sampling import SMOTE
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from fitter import Fitter, get_common_distributions
import warnings
warnings.filterwarnings('ignore')





"""--------------------------------------------data exploration/cleaning--------------------------------------------"""
directory = '/home/silver/PycharmProjects/AAEDRL/AndMal2020-dynamic-BeforeAndAfterReboot'
csv_files = [file for file in os.listdir(directory) if file.endswith(('before_reboot_Cat.csv', 'after_reboot_Cat.csv'))]

dfs = []
for file in csv_files:
    df = pd.read_csv(os.path.join(directory, file))
    reboot_status = "before" if "before" in file else "after"
    df["Before_or_After_Reboot"] = reboot_status
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df = df.drop(df.columns[df.nunique() == 1], axis = 1) # 47032, 129
df = df.drop(df.columns[df.nunique() == len(df)], axis = 1) # no change

# print(df.isna().sum().sum())
# print(df.isnull().sum().sum())
# print(df.isin([np.inf, -np.inf]).sum().sum())
datatypes = pd.DataFrame(df.dtypes)
df_count = df.count()
description = df.describe()
# description.to_csv('descriptive_stats.csv')
# df_datatypes.to_csv("dtypes.csv")
# df_null_count.to_csv("null.csv")


memory_cols = df.filter(regex='^Memory')
api_cols = df.filter(regex='^API')
network_cols = df.filter(regex='^Network')
battery_cols = df.filter(regex='^Battery')
logcat_cols = df.filter(regex='^Logcat')
process_col = df["Process_total"]


le = LabelEncoder()
cols_le = ["Category", "Family", "Hash", "Before_or_After_Reboot"]
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
df = cap_outliers_iqr(df, factor=40)


"""-----------------------------------------------data viz-----------------------------------------------"""
def histogram_plot(data, features):
    for i in features:
        plt.figure(figsize=(10, 10))
        plt.xticks(rotation=90, fontsize=15)
        plt.xlabel(f'{i}')
        plt.title(f"histogram for {i}")
        sns.histplot(data[i], bins=100, kde=True)
        # plt.savefig(f"{i}.png")

# histogram_plot(memory_cols, memory_cols.columns)
# histogram_plot(api_cols, api_cols.columns)
# histogram_plot(network_cols, network_cols.columns)
# histogram_plot(battery_cols, battery_cols.columns)
# histogram_plot(logcat_cols, logcat_cols.columns)

# cat_col = sns.countplot(df, x="Category")
# for p in cat_col.patches:
#     cat_col.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha='center', va='baseline', rotation = 45)

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

df = corr(df) # 34549, 112 => deleted 17 features

def remove_outliers_zscore(df, threshold=1.45):
    z_scores = np.abs(stats.zscore(df))
    df_cleaned = df[(z_scores < threshold).all(axis=1)]
    return df_cleaned

# threshold = np.linspace(1, 3, 10)
#
# remaining_points = [len(remove_outliers_zscore(df, t)) for t in threshold]
#
# plt.figure(figsize=(10, 6))
# plt.plot(threshold, remaining_points, marker='o')
# plt.title('Outlier elbow method')
# plt.xlabel('Z-Score Threshold')
# plt.ylabel('Number of Remaining Points')
# plt.grid(True)

# df_cl = remove_outliers_zscore(df)
# df_cl = df_cl.drop(df_cl.columns[df_cl.nunique() == 1], axis = 1)




def kolmogorov_smirnov_test(column, dist='norm'):
    D, p_value = stats.kstest(column, dist)
    return p_value > 0.05

def anderson_darling_test(column, dist='norm'):
    result = stats.anderson(column, dist=dist)
    return result.statistic < result.critical_values[2]

def fit_distributions(column):
    try:
        f = Fitter(column, distributions=get_common_distributions())
        f.fit()
        best_fit = f.get_best(method='sumsquare_error')
        return best_fit
    except Exception as e:
        return None




# lognorm: 10
# cauchy: 10
# norm: 2
# expon: 13
# gamma: 8
# rayleigh: 1
# exponpow: 59
# chi2: 1
# powerlaw: 1

"""-----------------------------------------------vertical data split-----------------------------------------------"""
y = df["Category"]
X = df.drop("Category", axis=1)

y = pd.get_dummies(y).astype(int)
cols = ['Backdoor', 'Trojan_Banker', 'PUA',
       'FileInfector', 'Ransomware', 'Trojan_Dropper', 'Trojan_SMS',
       'Trojan_Spy', 'Trojan', 'Adware', 'Riskware', 'Scareware']
y = y.rename(columns = dict(zip(y.columns, cols)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
def robust_scaler(df):
    scaler = RobustScaler()
    df = scaler.fit_transform(df)
    return df

X_train_rs = robust_scaler(X_train)
X_test_rs = robust_scaler(X_test)
y_train_np = y_train.to_numpy()
sm = SMOTE()
X_train_rs, y_train = sm.fit_resample(X_train_rs, y_train_np)

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

# for column in df.columns:
#     if kolmogorov_smirnov_test(df[column]):
#         print(f"Column {column} follows the specified distribution (Kolmogorov-Smirnov test).\n")
#     if anderson_darling_test(df[column]):
#         print(f"Column {column} follows the specified distribution (Anderson-Darling test).\n")
#     fit_distributions(df[column])
#     print("\n")





