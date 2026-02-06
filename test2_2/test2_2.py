# EW Correlation, lambd=0.94
import numpy as np
import pandas as pd

df = pd.read_csv("test2.csv")
lam = 0.94

X = df.values
T, n = X.shape

w = (1 - lam) * lam ** np.arange(T-1, -1, -1)
w = w / w.sum()

mu = (w[:, None] * X).sum(axis=0)

cov = sum(
    w[t] * np.outer(X[t] - mu, X[t] - mu)
    for t in range(T)
)
std = np.sqrt(np.diag(cov))

corr = cov / np.outer(std, std)
corr_df = pd.DataFrame(corr, columns=df.columns)
corr_df.to_csv("testout_2.2.csv", index=False)

print(corr_df)
