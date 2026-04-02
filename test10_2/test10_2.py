import numpy as np
import pandas as pd
from scipy.optimize import minimize

cov = pd.read_csv('test5_2.csv').values
n = len(cov)

# Risk budgets: equal for X1-X4, half weight for X5
budgets = np.array([1.0, 1.0, 1.0, 1.0, 0.5])
budgets /= budgets.sum()

def risk_budget_obj(w):
    w = np.array(w)
    sigma = np.sqrt(w @ cov @ w)
    rc = (cov @ w) * w / sigma
    return sum((rc[i]/budgets[i] - rc[j]/budgets[j])**2 for i in range(n) for j in range(n))

res = minimize(risk_budget_obj, np.ones(n)/n, method='SLSQP',
               bounds=[(0, None)]*n,
               constraints={'type': 'eq', 'fun': lambda w: np.sum(w)-1},
               options={'ftol': 1e-12, 'maxiter': 10000})

out = pd.DataFrame({'W': res.x})
out.to_csv('testout10_2.csv', index=False)
