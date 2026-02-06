# Covariance Missing data, Pairwise
import pandas as pd
import numpy as np
df = pd.read_csv("test1.csv")
df_clean = df.dropna()
covariance = df.cov(min_periods=1)
covariance.to_csv("testout_1.3.csv", index=True)