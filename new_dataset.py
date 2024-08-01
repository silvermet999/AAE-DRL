import pandas as pd
import main

synth = pd.read_csv("runs/rs38.csv")
df_synth = pd.DataFrame(synth)
# df_synth = df_synth.rename(columns = dict(zip(df_synth.columns, main.X.columns)))
X = pd.DataFrame(main.X_train_rs)
X = X.rename(columns = dict(zip(X.columns, main.X.columns)))
frames_un = [df_synth, X]
df_synth_plus = pd.concat(frames_un)