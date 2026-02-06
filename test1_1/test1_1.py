# Covariance Missing data, skip missing rows
import pandas as pd
import numpy as np
df = pd.read_csv("test1.csv")
df_clean = df.dropna()
covariance = df_clean.cov()
covariance.to_csv("testout_1.1.csv", index=True)

