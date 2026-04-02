import numpy as np
import pandas as pd

returns = pd.read_csv('test11_1_returns.csv').values
weights = pd.read_csv('test11_1_weights.csv').values.flatten()
T, n = returns.shape
assets = [f'x{i+1}' for i in range(n)]

# ── Evolve weights through time (buy-and-hold, no rebalancing) ──
w = weights.copy()
w_start = np.zeros((T, n))   # start-of-period weights
port_r  = np.zeros(T)
for t in range(T):
    w_start[t] = w
    w_star    = w * (1 + returns[t])
    port_r[t] = w_star.sum() - 1
    w         = w_star / (1 + port_r[t])

# ── Total Return ──
total_ret = np.prod(1 + returns, axis=0) - 1
port_total = np.prod(1 + port_r) - 1          # = weights @ total_ret

# ── Return Attribution via Cariño K ──
GR  = np.log(1 + port_total)
K   = GR / port_total
k_t = np.where(np.abs(port_r) < 1e-12, K, np.log(1 + port_r) / (K * port_r))
ret_attr = np.array([np.sum(k_t * w_start[:, i] * returns[:, i]) for i in range(n)])

# ── Vol Attribution: cov(w_i * r_i, portfolio) / sigma_p ──
port_std = np.std(port_r, ddof=1)
vol_attr = np.array([
    np.cov(w_start[:, i] * returns[:, i], port_r, ddof=1)[0, 1] / port_std
    for i in range(n)
])


rows = {
    'Value':       ['TotalReturn', 'Return Attribution', 'Vol Attribution'],
    **{assets[i]: [total_ret[i], ret_attr[i], vol_attr[i]] for i in range(n)},
    'Portfolio':   [port_total, ret_attr.sum(), vol_attr.sum()]
}
out = pd.DataFrame(rows)
out.to_csv('testout11_1.csv', index=False)
