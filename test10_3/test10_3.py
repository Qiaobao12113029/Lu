import numpy as np
import pandas as pd
from scipy.optimize import minimize

cov = pd.read_csv('test5_2.csv').values
mu  = pd.read_csv('test10_3_means.csv').values.flatten()
rf  = 0.04
n   = len(mu)

def neg_sharpe(w):
    w = np.array(w)
    return -(w @ mu - rf) / np.sqrt(w @ cov @ w)

res = minimize(neg_sharpe, np.ones(n)/n, method='SLSQP',
               bounds=[(0, None)]*n,
               constraints={'type': 'eq', 'fun': lambda w: np.sum(w)-1},
               options={'ftol': 1e-12, 'maxiter': 10000})

out = pd.DataFrame({'W': res.x})
out.to_csv('testout10_3.csv', index=False)
