# Normal Simulation PD Input 0 mean - 100,000 simulations, compare input vs output covariance
import numpy as np
import pandas as pd 

df = pd.read_csv("test5_1.csv")
df = df.select_dtypes(include=[np.number])

cov_input = df.cov().values
dim = cov_input.shape[0]

cov_input = (cov_input + cov_input.T) / 2

n_sim = 100000

L = np.linalg.cholesky(cov_input)

Z = np.random.normal(0, 1, size=(n_sim, dim))

simulated = Z @ L.T

cov_simulated = np.cov(simulated, rowvar=False)

cov_df = pd.DataFrame(cov_simulated, columns=df.columns)
cov_df.to_csv("testout_5.1.csv", index=False)
