import numpy as np
import pandas as pd

df = pd.read_csv("testout_1.4.csv")

A = df.values.astype(float)
A = (A + A.T) / 2
eigval, eigvec = np.linalg.eigh(A)
eigval[eigval < 0] = 0
A_psd = eigvec @ np.diag(eigval) @ eigvec.T

d = np.sqrt(np.diag(A_psd))
A_corr = A_psd / np.outer(d, d)
np.fill_diagonal(A_corr, 1.0)

out = pd.DataFrame(A_corr, columns=df.columns)
out.to_csv("testout_3.2.csv", index=False)

