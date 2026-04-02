import numpy as np
import pandas as pd
from scipy.optimize import minimize

cov = pd.read_csv('test5_2.csv').values
n = len(cov)

def risk_parity_obj(w):
    w = np.array(w)
    sigma = np.sqrt(w @ cov @ w)
    rc = (cov @ w) * w / sigma
    return sum((rc[i] - rc[j])**2 for i in range(n) for j in range(n))

res = minimize(risk_parity_obj, np.ones(n) / n, method='SLSQP',
               bounds=[(0, None)] * n,
               constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
               options={'ftol': 1e-12, 'maxiter': 10000})

out = pd.DataFrame({'W': res.x})
out.to_csv('testout10_1.csv', index=False)
