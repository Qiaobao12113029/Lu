# EW Covariance, lambda=0.97
import numpy as np
import pandas as pd

df = pd.read_csv("test2.csv", header=0)
lam = 0.97
X = df.values
T = X.shape[0]

w = np.array([(1 - lam) * lam ** (T - 1 - t) for t in range(T)], dtype=float)
w = w / w.sum()
mu = (w[:, None] * X).sum(axis=0)
n = X.shape[1]
cov = np.zeros((n, n), dtype=float)
for t in range(T):
    d = (X[t] - mu).reshape(-1, 1)
    cov += w[t] * (d @ d.T)


cov_df = pd.DataFrame(cov, index=df.columns, columns=df.columns)
cov_df.to_csv("testout_2.1.csv", index=False)