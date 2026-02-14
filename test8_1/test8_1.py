# Var from Normal Distribution\
import numpy as np
import pandas as pd
from math import fabs
from scipy.stats import norm

def compute_var_from_normal(data, alpha=0.95):
    x = np.asarray(data, dtype=float).ravel()
    mean = x.mean()
    s = x.std(ddof=1) 
    z = norm.ppf(alpha)

    var_diff = z * s
    var_abs = fabs(mean - var_diff)

    return var_abs, var_diff

def main():
    in_file = 'test7_1.csv'
    out_file = 'testout8_1.csv'

    df = pd.read_csv(in_file, header=0)
    if df.shape[1] < 1:
        raise RuntimeError("Input file must contain at least one column")

    data = df.iloc[:, 0].values

    var_abs, var_diff = compute_var_from_normal(data, alpha=0.95)

    out_df = pd.DataFrame({
        'VaR Absolute': [var_abs],
        'VaR Diff from Mean': [var_diff]
    })
    out_df.to_csv(out_file, index=False)


if __name__ == '__main__':
    main()