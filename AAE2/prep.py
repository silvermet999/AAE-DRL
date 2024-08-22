"""-----------------------------------------------import libraries-----------------------------------------------"""
import os

import torch
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

"""--------------------------------------------data exploration/cleaning--------------------------------------------"""
directory="C:\\Users\\professor\\AAE-DRL\\AndMal2020-dynamic-BeforeAndAfterReboot"

csv_files = [file for file in os.listdir(directory) if file.endswith(('before_reboot_Cat.csv', 'after_reboot_Cat.csv'))]

dfs = []
for file in csv_files:
    df = pd.read_csv(os.path.join(directory, file))
    reboot_status = "before" if "before" in file else "after"
    df["Before_or_After_Reboot"] = reboot_status
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df = df.drop(df.columns[df.nunique() == 1], axis = 1)
df = df.drop(df.columns[df.nunique() == len(df)], axis = 1) # no change

datatypes = pd.DataFrame(df.dtypes)
null_count = df.count()
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
    df = df[~df[col].isin(singletons)]


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

def plot_family():
    N = 50

    value_counts = df['Family'].value_counts()
    top_n = value_counts.nlargest(N)
    other_count = value_counts.sum() - top_n.sum()

    plot_data = pd.concat([top_n, pd.Series({'Other': other_count})])

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=plot_data.index, y=plot_data.values)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', rotation = 45)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {N} Categories')
    plt.tight_layout()

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

df = corr(df)

def remove_outliers_zscore(df, threshold=1.4):
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

"""-----------------------------------------------clf data split-----------------------------------------------"""
y = df["Category"]
X = df.drop("Category", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split(X_cl, y_cl, test_size=0.2, random_state=42)

"""-----------------------------------------------AAE data split-----------------------------------------------"""
cols = ['Backdoor', 'Trojan_Banker', 'Zero_Day', 'No_Category', 'PUA',
       'FileInfector', 'Ransomware', 'Trojan_Dropper', 'Trojan_SMS',
       'Trojan_Spy', 'Trojan', 'Adware', 'Riskware', 'Scareware']
df_h = pd.get_dummies(df, columns = ["Category"]).astype(int)
df_h = df_h.rename(columns = dict(zip(df_h.filter(regex="^Category"), cols)))

train = df_h[:31795]
test = df_h.iloc[31795:]


# y_cl = df_cl["Category"]
# X_cl = df_cl.drop("Category", axis = 1)


"""-----------------------------------------------data preprocessing-----------------------------------------------"""
def robust_scaler(df):
    scaler = RobustScaler()
    df = scaler.fit_transform(df)
    return df

X_train_rs = RobustScaler()


def max_abs_scaler(df):
    scaler = MaxAbsScaler()
    df = scaler.fit_transform(df)
    return df
