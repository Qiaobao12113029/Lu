import numpy as np
import pandas as pd
from scipy.stats import norm, t, multivariate_normal, rankdata

returns_file = "test9_1_returns.csv"
portfolio_file = "test9_1_portfolio.csv"
output_file = "testout9_1.csv"
alpha = 0.95
n_sims = 20000
seed = 2026
np.random.seed(seed)

rets = pd.read_csv(returns_file)
pf = pd.read_csv(portfolio_file)
assets = list(rets.columns)
m = len(assets)
T = len(rets)

holdings = {}
prices = {}
dists = {}
for _, row in pf.iterrows():
    stock = str(row['Stock'])
    holdings[stock] = float(row['Holding'])
    prices[stock] = float(row['Starting Price'])
    dists[stock] = str(row['Distribution']).strip().lower()

marginals = {} 
for col in assets:
    arr = rets[col].dropna().values.astype(float)
    if dists.get(col, "normal").startswith("t"):
        df_fit, loc_fit, scale_fit = t.fit(arr)
        marginals[col] = {
            "type": "t",
            "df": df_fit,
            "loc": loc_fit,
            "scale": scale_fit
        }
    else:
        mu = arr.mean()
        sigma = arr.std(ddof=1)
        marginals[col] = {
            "type": "normal",
            "loc": mu,
            "scale": sigma
        }

rows = []
for col in assets:
    params = marginals[col]
    position_value = holdings[col] * prices[col]
    if params['type'] == 'normal':
        mu = params['loc']; sigma = params['scale']
        z = norm.ppf(1 - alpha)
        VaR_pct = abs(mu + sigma * z)
        ES_pct = abs(mu - sigma * (norm.pdf(z) / (1 - alpha)))
    else:
        df_fit = params['df']; loc = params['loc']; scale = params['scale']
        q = t.ppf(1 - alpha, df_fit)
        ES_val = loc - scale * ((df_fit + q**2) / ((df_fit - 1) * (1 - alpha))) * t.pdf(q, df_fit)
        VaR_val = loc + scale * q
        VaR_pct = abs(VaR_val)
        ES_pct = abs(ES_val)

    VaR_amt = position_value * VaR_pct
    ES_amt = position_value * ES_pct

    rows.append({
        "Stock": col,
        "VaR95": VaR_amt,
        "ES95": ES_amt,
        "VaR95_Pct": VaR_pct,
        "ES95_Pct": ES_pct
    })

Z = np.zeros((T, m))
for j, col in enumerate(assets):
    vals = rets[col].dropna().values.astype(float)
    ranks = rankdata(vals, method='average')
    u = ranks / (len(vals) + 1.0)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    Z[:, j] = norm.ppf(u)

R = np.corrcoef(Z, rowvar=False)
eps = 1e-12
try:
    np.linalg.cholesky(R)
except np.linalg.LinAlgError:
    R = R + np.eye(m) * eps

mv = multivariate_normal(mean=np.zeros(m), cov=R)

z_sim = mv.rvs(size=n_sims)
u_sim = norm.cdf(z_sim)
u_sim = np.clip(u_sim, 1e-12, 1 - 1e-12)

sim_returns = np.zeros_like(u_sim)
for j, col in enumerate(assets):
    params = marginals[col]
    if params['type'] == 'normal':
        sim_returns[:, j] = norm.ppf(u_sim[:, j], loc=params['loc'], scale=params['scale'])
    else:
        sim_returns[:, j] = t.ppf(u_sim[:, j], params['df'], loc=params['loc'], scale=params['scale'])

pos_values = np.array([holdings[a] * prices[a] for a in assets])
loss_sim = - (sim_returns * pos_values).sum(axis=1)

VaR_port_95 = np.quantile(loss_sim, alpha)
ES_port_95 = loss_sim[loss_sim >= VaR_port_95].mean()

total_value = pos_values.sum()
VaR_port_95_pct = VaR_port_95 / total_value
ES_port_95_pct = ES_port_95 / total_value

rows.append({
    "Stock": "Total",
    "VaR95": VaR_port_95,
    "ES95": ES_port_95,
    "VaR95_Pct": VaR_port_95_pct,
    "ES95_Pct": ES_port_95_pct
})

out_df = pd.DataFrame(rows, columns=["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"])
out_df.to_csv(output_file, index=False)
