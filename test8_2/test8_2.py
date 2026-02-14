# Var from T Distribution
import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from math import fabs

def compute_var_from_t(data, alpha=0.95):
    x = np.asarray(data, dtype=float).ravel()
    n = x.size
    if n < 2:
        raise ValueError("Need at least 2 observations for t-based VaR")
    mean = x.mean()
    s = x.std(ddof=1) 
    df = n - 1
    t_q = tdist.ppf(alpha, df)
    var_diff = t_q * s
    var_abs = fabs(mean - var_diff)
    return var_abs, var_diff

def main():
    in_file = 'test7_2.csv'
    out_file = 'testout8_2.csv'

    df = pd.read_csv(in_file, header=0)
    data = df.iloc[:, 0].values

    var_abs, var_diff = compute_var_from_t(data, alpha=0.95)

    out_df = pd.DataFrame({
        'VaR Absolute': [var_abs],
        'VaR Diff from Mean': [var_diff]
    })

    out_df.to_csv(out_file, index=False)

if __name__ == '__main__':
    main()