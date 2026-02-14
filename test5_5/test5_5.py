# PCA Simulation, 99% explained, 0 mean - 100,000 simulations compare input vs output covariance
import numpy as np
import pandas as pd

def pca_simulation_cov(A, explained=0.99, n_sim=100000, seed=12345):
    """
    Given covariance matrix A, perform PCA, keep enough components to explain
    'explained' fraction of total variance, simulate n_sim draws from N(0,A)
    using the low-rank PCA representation, and return the sample covariance.
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    idx = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[idx]
    eigvecs_sorted = eigvecs[:, idx]

    total_var = eigvals_sorted.sum()
    if total_var <= 0:
        raise ValueError("Total variance is non-positive; check input matrix.")

    cumvar = np.cumsum(eigvals_sorted) / total_var
    k = int(np.searchsorted(cumvar, explained)) + 1

    vals_keep = eigvals_sorted[:k].copy()
    vecs_keep = eigvecs_sorted[:, :k].copy()

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n_sim, k))
    transform = vecs_keep * np.sqrt(vals_keep)[np.newaxis, :]  # p x k
    samples = Z @ transform.T

    sample_cov = (samples.T @ samples) / n_sim

    return sample_cov, k

def main():
    in_file = 'test5_2.csv'
    out_file = 'testout_5.5.csv'

    df = pd.read_csv(in_file, header=0)
    A = df.values.astype(float)
    if A.shape[0] != A.shape[1]:
        raise RuntimeError("Input matrix must be square. Check test5_2.csv format.")

    sample_cov, k = pca_simulation_cov(A, explained=0.99, n_sim=100000, seed=12345)

    pd.DataFrame(sample_cov, index=df.columns, columns=df.columns).to_csv(out_file, header=True, index=False)


if __name__ == '__main__':
    main()