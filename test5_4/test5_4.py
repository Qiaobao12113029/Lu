# Normal Simulation PSD Input, 0 mean, higham fix - 100,000 simulations, compare input vs output covariance

import numpy as np
import pandas as pd

def near_psd(A, tol=1e-10, max_iter=100):
    """
    Higham (2002) algorithm to find the nearest symmetric positive semidefinite matrix.
    Returns a symmetric PSD matrix.
    """
    B = (A + A.T) / 2.0
    X = B.copy()
    Y = np.zeros_like(A)
    for k in range(max_iter):
        R = X - Y
        R = (R + R.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals_clipped = np.clip(eigvals, 0, None)   # set negative eigenvalues to zero
        X_new = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        Y = X_new - R
        X = X_new
        if np.linalg.norm(X - B, ord='fro') < tol:
            break
    return (X + X.T) / 2.0

def sample_mvnormal_zero(cov, n, seed=12345):
    """
    Draw n samples from N(0, cov). If cov is not cholesky-able (singular),
    use eigendecomposition to generate samples.
    """
    cov = (cov + cov.T) / 2.0
    rng = np.random.default_rng(seed)
    try:
        L = np.linalg.cholesky(cov)
        z = rng.standard_normal(size=(n, cov.shape[0]))
        return z @ L.T
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        vals[vals < 0] = 0.0
        sqrt_vals = np.sqrt(vals)
        transform = vecs * sqrt_vals[np.newaxis, :]
        z = rng.standard_normal(size=(n, cov.shape[0]))
        return z @ transform.T

def main():
    in_file = 'test5_3.csv'
    out_file = 'testout_5.4.csv'
    df = pd.read_csv(in_file, header=0)
    A = df.values.astype(float)

    A_psd = near_psd(A)

    n_sim = 100000
    samps = sample_mvnormal_zero(A_psd, n_sim, seed=12345)

    sample_cov = (samps.T @ samps) / samps.shape[0]
    pd.DataFrame(sample_cov, index=df.columns, columns=df.columns).to_csv(out_file, index=False)

if __name__ == '__main__':
    main()