import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.optimize import minimize

data = pd.read_csv("test7_3.csv")
y = data["y"].to_numpy(dtype=float)
X = data[["x1", "x2", "x3"]].to_numpy(dtype=float)
n = len(y)

def nll(p):
    a, b1, b2, b3, s, nu = p
    if s <= 0 or nu <= 2:
        return np.inf
    m = a + X @ np.array([b1, b2, b3])
    return -np.sum(t.logpdf(y, df=nu, loc=m, scale=s))

Xd = np.column_stack([np.ones(n), X])
b0 = np.linalg.lstsq(Xd, y, rcond=None)[0]

p0 = np.array([b0[0], b0[1], b0[2], b0[3], np.std(y, ddof=0), 10.0], dtype=float)

res = minimize(
    nll,
    p0,
    method="L-BFGS-B",
    bounds=[(None, None), (None, None), (None, None), (None, None), (1e-12, None), (2.01, 1e6)],
    options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 200000}
)

a, b1, b2, b3, s, nu = res.x

out = pd.DataFrame({
    "mu": [0.0],
    "sigma": [float(s)],
    "nu": [float(nu)],
    "Alpha": [float(a)],
    "B1": [float(b1)],
    "B2": [float(b2)],
    "B3": [float(b3)]
})

out.to_csv("testout7_3.csv", index=False, float_format="%.15f")

