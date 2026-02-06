import numpy as np
import pandas as pd

df = pd.read_csv("testout_1.3.csv")
A = df.values.astype(float)

A = (A + A.T) / 2
eigval, eigvec = np.linalg.eigh(A)
eigval[eigval < 0] = 0
A_psd = eigvec @ np.diag(eigval) @ eigvec.T
A_psd = (A_psd + A_psd.T) / 2

out = pd.DataFrame(A_psd, columns=df.columns)
out.to_csv("testout_3.1.csv", index=False)

