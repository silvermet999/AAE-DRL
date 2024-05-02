import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



directory = 'AndMal2020-dynamic-BeforeAndAfterReboot'
csv_files = [file for file in os.listdir(directory) if file.endswith(('before_reboot_Cat.csv', 'after_reboot_Cat.csv'))]

dfs = []
for file in csv_files:
    df = pd.read_csv(os.path.join(directory, file))
    reboot_status = "before" if "before" in file else "after"
    df["Before or After reboot"] = reboot_status
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df.drop("Hash", axis=1)



def univariate_analysis(data, features):
    for feature in features:
        plt.figure(figsize=(10, 5))
        plt.xticks(rotation=90, fontsize=15)
        plt.xlabel(f'{feature}')
        plt.title(f"Bar plot for {feature}")
        sns.histplot(data[feature], bins=100, kde=False)
        plt.savefig(f"{feature}.png")
        plt.show()

memory_columns = df.filter(regex='^Memory')
api_columns = df.filter(regex='^API')
network_columns = df.filter(regex='^Network')
battery_columns = df.filter(regex='^Battery')
logcat_columns = df.filter(regex='^Logcat')
process_column = df["Process_total"]

# univariate_analysis(memory_columns, memory_columns.columns)
# univariate_analysis(api_columns, api_columns.columns)
# univariate_analysis(network_columns, network_columns.columns)
# univariate_analysis(battery_columns, battery_columns.columns)
# univariate_analysis(logcat_columns, logcat_columns.columns)

plt.figure(figsize=(20, 20))
sns.heatmap(memory_columns.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
# plt.savefig("memory corr")



def multivariate_analysis(data_col, dfs):
    sns.set_theme(style="ticks")
    for i in data_col.columns:
        for df in dfs:
            sns.lmplot(data=df, x=data_col[i], col="Before or After reboot", hue="Before or After reboot",
                       col_wrap=2, palette="muted", ci=None, height=4, scatter_kws={"s": 50, "alpha": 1})
