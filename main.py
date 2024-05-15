"""-----------------------------------------------import libraries-----------------------------------------------"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA



"""--------------------------------------------data exploration/cleaning--------------------------------------------"""
directory = 'AndMal2020-dynamic-BeforeAndAfterReboot'
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


def multivariate_analysis(data_col):
    sns.set_theme(style="ticks")
    for i in data_col.columns:
        for df in dfs:
            sns.lmplot(data=df, x=data_col.loc[i], col="Before_or_After_Reboot", hue="Before_or_After_Reboot",
                       col_wrap=2, palette="muted", ci=None, height=4, scatter_kws={"s": 50, "alpha": 1})


encoder = LabelEncoder()
cols = ["Category", "Family", "Before_or_After_Reboot", "Hash"]
for i in cols:
    df[i] = encoder.fit_transform(df[i])


correlation = df.corr()
f_corr = {}
for column in correlation.columns:
    correlated_with = list(correlation.index[correlation[column] >= 0.75])
    for corr_col in correlated_with:
        if corr_col != column:
            df_corr = correlation.loc[corr_col, column]
            f_corr[(column, corr_col)] = df_corr
f_corr = pd.DataFrame.from_dict(f_corr, orient="index")
f_corr = f_corr.drop_duplicates()
# f_corr.to_csv("f_corr.csv")
df_no_corr_3 = df.drop(df_corr[["Memory_PssClean", "Memory_HeapAlloc", "Memory_HeapFree",
                                "API_Binder_android.app.ContextImpl_registerReceiver",
                                "API_DexClassLoader_dalvik.system.BaseDexClassLoader_findLibrary",
                                "Network_TotalReceivedBytes", "Network_TotalReceivedPackets"]], axis = 1)
df_no_corr_2 = df_no_corr_3.drop(df_no_corr_3[["Memory_PrivateDirty", "Memory_Activities", "Memory_ProxyBinders",
                                   "Memory_ParcelMemory", "API_Command_java.lang.ProcessBuilder_start",
                                   "API_Database_android.database.sqlite.SQLiteDatabase_deleteDatabase",
                                               "API_Database_android.database.sqlite.SQLiteDatabase_getPath",
                                               "API_Database_android.database.sqlite.SQLiteDatabase_compileStatement",
                                               "API_Database_android.database.sqlite.SQLiteDatabase_query",
                                               "API_IPC_android.content.ContextWrapper_startActivity",
                                               "API_DeviceInfo_android.net.wifi.WifiInfo_getBSSID",
                                               "API_DeviceInfo_android.net.wifi.WifiInfo_getIpAddress",
                                               "API_Base64_android.util.Base64_encode",
                                               "API_DeviceData_android.location.Location_getLatitude",
                                               "Logcat_error"]], axis = 1)



"""-----------------------------------------------vertical data split-----------------------------------------------"""
y = df[["Category", "Family", "Before_or_After_Reboot"]]
X = df.drop(y, axis = 1)



"""-----------------------------------------------data preprocessing-----------------------------------------------"""
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


def PCA_alg():
    pca = PCA(n_components=100)


