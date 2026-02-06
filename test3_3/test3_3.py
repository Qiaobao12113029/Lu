import numpy as np
import pandas as pd

def near_pd_higham(A, preserve_diag=True, tol=1e-8, max_iter=200):
    """
    Higham-like nearPD with Dykstra-style correction.
    """
    n = A.shape[0]
    A = (A + A.T) / 2.0
    X = A.copy()
    deltaS = np.zeros_like(A)
    orig_diag = np.diag(A).copy()
    for k in range(1, max_iter + 1):
        R = X - deltaS
        R = (R + R.T) / 2.0
        vals, vecs = np.linalg.eigh(R)
        vals_clipped = np.where(vals > 0, vals, 0.0)
        X_new = (vecs * vals_clipped) @ vecs.T
        X_new = (X_new + X_new.T) / 2.0
        deltaS = X_new - R
        if preserve_diag:
            np.fill_diagonal(X_new, orig_diag)
        diff = np.linalg.norm(X_new - X, ord='fro')
        X = X_new
        if diff < tol:
            break
    X = (X + X.T) / 2.0
    vals, vecs = np.linalg.eigh(X)
    vals[vals < 0] = 0.0
    X = (vecs * vals) @ vecs.T
    X = (X + X.T) / 2.0
    return X, k

if __name__ == "__main__":
    df = pd.read_csv("testout_1.3.csv")
    
    cols = df.columns.copy()
    A = df.values.astype(float)
    A_pd, iters = near_pd_higham(A, preserve_diag=True, tol=1e-9, max_iter=500)

    out_df = pd.DataFrame(A_pd, columns=cols)
    out_df.to_csv( "testout_3.3.csv", index=False)

