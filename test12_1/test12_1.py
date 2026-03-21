import numpy as np
import pandas as pd
from scipy.stats import norm

def gbsm(option_type, S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = norm.pdf(d1)

    if option_type.strip().lower() == 'call':
        value = S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        delta = np.exp(-q*T) * norm.cdf(d1)
        rho   = K * T * np.exp(-r*T) * norm.cdf(d2)
        theta = (-(S * sigma * np.exp(-q*T) * nd1) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r*T) * norm.cdf(d2)
                 + q * S * np.exp(-q*T) * norm.cdf(d1))
    else:
        value = K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)
        delta = -np.exp(-q*T) * norm.cdf(-d1)
        rho   = -K * T * np.exp(-r*T) * norm.cdf(-d2)
        theta = (-(S * sigma * np.exp(-q*T) * nd1) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r*T) * norm.cdf(-d2)
                 - q * S * np.exp(-q*T) * norm.cdf(-d1))

    gamma = np.exp(-q*T) * nd1 / (S * sigma * np.sqrt(T))
    vega  = S * np.exp(-q*T) * nd1 * np.sqrt(T)

    return value, delta, gamma, vega, rho, theta

df = pd.read_csv('test12_1.csv').dropna(how='all')

rows = []
for _, r in df.iterrows():
    T = r['DaysToMaturity'] / r['DayPerYear']
    res = gbsm(r['Option Type'], r['Underlying'], r['Strike'], T,
               r['RiskFreeRate'], r['DividendRate'], r['ImpliedVol'])
    rows.append([r['ID']] + list(res))

out = pd.DataFrame(rows, columns=['ID','Value','Delta','Gamma','Vega','Rho','Theta'])
out.to_csv('testout12_1.csv', index=False)
