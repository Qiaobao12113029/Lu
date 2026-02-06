# Correlation Missing data, skip missing rows
import pandas as pd
import numpy as np
df = pd.read_csv("test1.csv")
df_clean = df.dropna()
correlation = df_clean.corr()
correlation.to_csv("testout_1.2.csv", index=True)