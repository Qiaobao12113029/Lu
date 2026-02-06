# Correlation Missing data, pairwise
import pandas as pd
import numpy as np
df = pd.read_csv("test1.csv")
corr_matrix = df.corr()
corr_matrix.to_csv('testout_1.4.csv', index=True)