import numpy as np
import pandas as pd

def ew_cov_weighted_sum(X, lam):
    T = X.shape[0]
    w = (1 - lam) * lam ** np.arange(T - 1, -1, -1)
    w = w / w.sum()
    mu = (w[:, None] * X).sum(axis=0)
    cov = sum(w[t] * np.outer(X[t] - mu, X[t] - mu) for t in range(T))
    return cov

df = pd.read_csv("test2.csv")
X = df.values
lam_var = 0.97
lam_corr = 0.94

cov_for_var = ew_cov_weighted_sum(X, lam_var)
var_ew = np.diag(cov_for_var)

cov_for_corr = ew_cov_weighted_sum(X, lam_corr)
std_corr = np.sqrt(np.diag(cov_for_corr))

std_corr[std_corr == 0] = np.nan
corr_ew = cov_for_corr / np.outer(std_corr, std_corr)
n = X.shape[1]
M = np.zeros((n, n), dtype=float)
for i in range(n):
    for j in range(n):
        if i == j:
            M[i, j] = var_ew[i]
        else:
            M[i, j] = corr_ew[i, j] * np.sqrt(var_ew[i] * var_ew[j])

out_df = pd.DataFrame(M, columns=df.columns)

out_df.to_csv("testout_2.3.csv", index=False, float_format="%.16f")

print(out_df)
