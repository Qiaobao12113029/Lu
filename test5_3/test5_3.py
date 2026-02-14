# Normal Simulation nonPSD Input, 0 mean, near_psd fix - 100,000 simulations, compare input vs output covariance
import numpy as np
import pandas as pd

np.random.seed(123)
N_SIMS = 100_000
NEAR_PSD_EPS = 1e-12

def nearest_psd_by_eig(A, tol=1e-12):
    A_sym = (A + A.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    eigvals_clipped = np.where(eigvals < tol, tol, eigvals)
    A_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    A_psd = (A_psd + A_psd.T) / 2.0
    return A_psd, eigvecs, eigvals_clipped

def frobenius_norm(X):
    return np.sqrt(np.sum((X)**2))

df = pd.read_csv("test5_3.csv")
df_num = df.select_dtypes(include=[np.number]).copy()
cols = list(df_num.columns)
cov_input = df_num.cov().values
cov_input = (cov_input + cov_input.T) / 2.0
cov_psd, eigvecs, eigvals_clipped = nearest_psd_by_eig(cov_input, tol=NEAR_PSD_EPS)

sqrt_diag = np.sqrt(eigvals_clipped)
sqrt_cov = eigvecs @ np.diag(sqrt_diag)

dim = cov_psd.shape[0]
Z = np.random.normal(loc=0.0, scale=1.0, size=(N_SIMS, dim))
simulated = Z @ sqrt_cov.T

cov_simulated = np.cov(simulated, rowvar=False)
cov_simulated = (cov_simulated + cov_simulated.T) / 2.0

cov_df = pd.DataFrame(cov_simulated, columns=cols)
cov_df.to_csv("testout_5.3.csv", index=False)
