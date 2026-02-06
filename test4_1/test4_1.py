import numpy as np
import pandas as pd

def eigclip_to_psd(A, eps=0.0):
    """Symmetrize and clip negative eigenvalues to eps (default 0.0)."""
    A = (A + A.T) / 2.0
    vals, vecs = np.linalg.eigh(A)
    vals_clipped = np.where(vals > eps, vals, eps)
    A_psd = (vecs * vals_clipped) @ vecs.T
    return (A_psd + A_psd.T) / 2.0

if __name__ == "__main__":
    df = pd.read_csv("testout_3.1.csv")

    cols = df.columns.copy()
    A = df.values.astype(float)
    A = (A + A.T) / 2.0

    vals = np.linalg.eigvalsh(A)
    if vals.min() < -1e-12:
        A = eigclip_to_psd(A, eps=0.0)
    jitter = 0.0
    max_tries = 20
    success = False
    for k in range(max_tries):
        try:
            if jitter > 0:
                A_try = A + np.eye(A.shape[0]) * jitter
            else:
                A_try = A
            L = np.linalg.cholesky(A_try)
            success = True
            break
        except np.linalg.LinAlgError:
            jitter = 10**(-12 + k) if jitter == 0 else jitter * 10
    if not success:
        A = eigclip_to_psd(A, eps=1e-12)
        L = np.linalg.cholesky(A + np.eye(A.shape[0]) * 1e-14)

    L_df = pd.DataFrame(L, columns=cols)
    L_df.to_csv("testout_4.1.csv", index=False)