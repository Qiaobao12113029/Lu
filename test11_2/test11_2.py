import numpy as np
import pandas as pd

stock_ret  = pd.read_csv('test11_2_stock_returns.csv').values
factor_ret = pd.read_csv('test11_2_factor_returns.csv').values
beta       = pd.read_csv('test11_2_beta.csv').iloc[:, 1:].values.astype(float)
weights    = pd.read_csv('test11_2_weights.csv').values.flatten()
T, n_stocks  = stock_ret.shape
n_factors    = factor_ret.shape[1]
factor_names = pd.read_csv('test11_2_factor_returns.csv').columns.tolist()

# ── Evolve stock weights through time (buy-and-hold) ──
w = weights.copy()
w_start = np.zeros((T, n_stocks))
port_r  = np.zeros(T)
for t in range(T):
    w_start[t] = w
    w_star     = w * (1 + stock_ret[t])
    port_r[t]  = w_star.sum() - 1
    w          = w_star / (1 + port_r[t])

# Factor weights at each period: w_factor_k,t = sum_i w_i,t * beta_i,k
w_factor = w_start @ beta                              # (T, n_factors)

# Alpha per period = portfolio return - factor-explained return
alpha_r  = port_r - (w_factor * factor_ret).sum(axis=1)

# ── Total Returns ──
total_factor = np.prod(1 + factor_ret, axis=0) - 1
port_total   = np.prod(1 + port_r) - 1
alpha_total  = port_total - (weights @ beta) @ total_factor

# ── Cariño K scaling ──
GR  = np.log(1 + port_total)
K   = GR / port_total
k_t = np.where(np.abs(port_r) < 1e-12, K, np.log(1 + port_r) / (K * port_r))

# ── Return Attribution ──
ret_attr_f = np.array([np.sum(k_t * w_factor[:, k] * factor_ret[:, k]) for k in range(n_factors)])
ret_attr_a = np.sum(k_t * alpha_r)

# ── Vol Attribution: cov(component, portfolio) / sigma_p ──
port_std   = np.std(port_r, ddof=1)
vol_attr_f = np.array([
    np.cov(w_factor[:, k] * factor_ret[:, k], port_r, ddof=1)[0, 1] / port_std
    for k in range(n_factors)
])
vol_attr_a = np.cov(alpha_r, port_r, ddof=1)[0, 1] / port_std

# ── Assemble output ──
cols = factor_names + ['Alpha', 'Portfolio']
data = {
    'Value': ['TotalReturn', 'Return Attribution', 'Vol Attribution'],
    **{factor_names[k]: [total_factor[k], ret_attr_f[k], vol_attr_f[k]] for k in range(n_factors)},
    'Alpha':     [alpha_total,  ret_attr_a,  vol_attr_a],
    'Portfolio': [port_total,   ret_attr_f.sum() + ret_attr_a, vol_attr_f.sum() + vol_attr_a],
}
out = pd.DataFrame(data)
out.to_csv('testout11_2.csv', index=False)
