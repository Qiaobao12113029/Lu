# Normal Simulation PSD Input 0 mean - 100,000 simulations, compare input vs output covariance

import numpy as np
import pandas as pd

df = pd.read_csv("test5_2.csv")
df = df.select_dtypes(include=[np.number])

cov_input = df.cov().values
cov_input = (cov_input + cov_input.T) / 2 

dim = cov_input.shape[0]
n_sim = 100000

eigvals, eigvecs = np.linalg.eigh(cov_input)

eigvals[eigvals < 0] = 0

sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals))

Z = np.random.normal(0, 1, size=(n_sim, dim))

simulated = Z @ sqrt_cov.T

cov_simulated = np.cov(simulated, rowvar=False)

cov_df = pd.DataFrame(cov_simulated, columns=df.columns)
cov_df.to_csv("testout_5.2.csv", index=False)
