import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler, MaxAbsScaler



directory = 'AndMal2020-dynamic-BeforeAndAfterReboot'
csv_files = [file for file in os.listdir(directory) if file.endswith(('before_reboot_Cat.csv', 'after_reboot_Cat.csv'))]

dfs = []
for file in csv_files:
    df = pd.read_csv(os.path.join(directory, file))
    reboot_status = "before" if "before" in file else "after"
    df["Before or After reboot"] = reboot_status
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df = df.drop(["Hash"], axis=1)
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


def corr_map(cols):
    plt.figure(figsize=(20, 20))
    sns.heatmap(cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    # plt.savefig("memory corr")



def univariate_analysis(data, features):
    for i in features:
        plt.figure(figsize=(10, 10))
        plt.xticks(rotation=90, fontsize=15)
        plt.xlabel(f'{i}')
        plt.title(f"Bar plot for {i}")
        sns.histplot(data[i], bins=100, kde=False)
        # plt.savefig(f"{i}.png")

# univariate_analysis(memory_cols, memory_cols.columns)
# univariate_analysis(api_cols, api_cols.columns)
# univariate_analysis(network_cols, network_cols.columns)
# univariate_analysis(battery_cols, battery_cols.columns)
# univariate_analysis(logcat_cols, logcat_cols.columns)



def multivariate_analysis(data_col):
    sns.set_theme(style="ticks")
    for i in data_col.columns:
        for df in dfs:
            sns.lmplot(data=df, x=data_col.loc[i], col="Before or After reboot", hue="Before or After reboot",
                       col_wrap=2, palette="muted", ci=None, height=4, scatter_kws={"s": 50, "alpha": 1})



encoder = LabelEncoder()
y = ["Category", "Family", "Before or After reboot"]
for i in y:
    df[i] = encoder.fit_transform(df[i])


X = df.drop(y, axis = 1)

def robust_scaler(df):
    scaler = RobustScaler()
    df = scaler.fit_transform(df)
    return df

X_rs = robust_scaler(X)


def max_abs_scaler(df):
    scaler = MaxAbsScaler()
    df = scaler.fit_transform(df)
    return df

X_mas = max_abs_scaler(X)