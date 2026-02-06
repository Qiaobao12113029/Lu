import numpy as np
import pandas as pd

def project_to_psd(A):
    """Symmetric eigen-clipping PSD projection."""
    A = (A + A.T) / 2.0
    vals, vecs = np.linalg.eigh(A)
    vals_clipped = np.clip(vals, 0.0, None)
    A_psd = (vecs * vals_clipped) @ vecs.T
    return (A_psd + A_psd.T) / 2.0

def near_pd_correlation_higham(A, max_iter=500, tol=1e-10):
    """
    Higham-like nearest correlation matrix (Dykstra-style correction).
    """
    n = A.shape[0]
    A0 = (A + A.T) / 2.0
    Y = A0.copy()
    deltaS = np.zeros_like(A0)
    prev_Y = Y.copy()

    for k in range(1, max_iter+1):
        R = Y - deltaS
        X = project_to_psd(R)
        deltaS = X - R
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)
        diff = np.linalg.norm(Y - prev_Y, ord='fro')
        if diff < tol:
            return ( (Y + Y.T)/2.0, k )
        prev_Y = Y.copy()
    Y = project_to_psd(Y)
    np.fill_diagonal(Y, 1.0)
    Y = (Y + Y.T) / 2.0
    return (Y, max_iter)

if __name__ == "__main__":
    df = pd.read_csv("testout_1.4.csv", header=0)
    cols = df.columns.copy()
    A = df.values.astype(float)
    A_corr, iters = near_pd_correlation_higham(A, max_iter=1000, tol=1e-11)

    A_corr = (A_corr + A_corr.T) / 2.0
    np.fill_diagonal(A_corr, 1.0)

    out_df = pd.DataFrame(A_corr, columns=cols)
    out_df.to_csv("testout_3.4.csv", index=False)

