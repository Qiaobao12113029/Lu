import numpy as np
import pandas as pd
from scipy.stats import t
import os

input_file = "test7_2.csv"
output_file = "testout8_6.csv"
alpha = 0.95
n_sims = 100000 
seed = 2026

df = pd.read_csv(input_file)
returns = df['x1'].astype(float).dropna().values

df_fit, loc_fit, scale_fit = t.fit(returns)

q = t.ppf(1 - alpha, df_fit)
ES_analytic = loc_fit - scale_fit * ((df_fit + q**2) / ((df_fit - 1) * (1 - alpha))) * t.pdf(q, df_fit)

np.random.seed(seed)
sim_samples = t.rvs(df_fit, loc=loc_fit, scale=scale_fit, size=n_sims)

VaR_sim = np.quantile(sim_samples, 1 - alpha)
tail = sim_samples[sim_samples <= VaR_sim]
ES_sim = tail.mean() if tail.size > 0 else float('nan')

ES_absolute = abs(ES_sim)
ES_diff_from_mean = loc_fit - ES_sim

out_df = pd.DataFrame({
    "ES Absolute": [ES_absolute],
    "ES Diff from Mean": [ES_diff_from_mean]
})

out_df.to_csv(output_file, index=False)
