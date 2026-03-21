import numpy as np
import pandas as pd

def american_binomial(option_type, S, K, T, r, q, sigma, N=500):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(N + 1)
    ST = S * u**j * d**(N - j)
    V = np.maximum(ST - K, 0) if option_type.lower() == 'call' else np.maximum(K - ST, 0)
    for i in range(N - 1, -1, -1):
        ST_i = S * u**np.arange(i + 1) * d**(i - np.arange(i + 1))
        V = disc * (p * V[1:i+2] + (1 - p) * V[0:i+1])
        V = np.maximum(V, ST_i - K) if option_type.lower() == 'call' else np.maximum(V, K - ST_i)
        if i == 2: V2, ST2 = V.copy(), ST_i.copy()
        if i == 1: V1, ST1 = V.copy(), ST_i.copy()
    return V[0], V1, ST1, V2, ST2

def greeks(option_type, S, K, T, r, q, sigma):
    p0, V1, ST1, V2, ST2 = american_binomial(option_type, S, K, T, r, q, sigma)
    pu, *_ = american_binomial(option_type, S+0.001, K, T, r, q, sigma)
    pd, *_ = american_binomial(option_type, S-0.001, K, T, r, q, sigma)
    pt, *_ = american_binomial(option_type, S, K, T-1/365, r, q, sigma)
    pv, *_ = american_binomial(option_type, S, K, T, r, q, sigma+0.001)
    pr, *_ = american_binomial(option_type, S, K, T, r+0.001, q+0.001, sigma)

    delta = (pu - pd) / 0.002
    d1_ = (V2[2]-V2[1])/(ST2[2]-ST2[1])
    d2_ = (V2[1]-V2[0])/(ST2[1]-ST2[0])
    gamma = (d1_ - d2_) / (0.5*(ST2[2]-ST2[0]))
    vega  = (pv - p0) / 0.001
    rho   = (pr - p0) / 0.001
    theta = abs((pt - p0) * 365)
    return p0, delta, gamma, vega, rho, theta

df = pd.read_csv('test12_1.csv').dropna(how='all')
rows = []
for _, row in df.iterrows():
    T = row['DaysToMaturity'] / row['DayPerYear']
    res = greeks(row['Option Type'], row['Underlying'], row['Strike'], T,
                 row['RiskFreeRate'], row['DividendRate'], row['ImpliedVol'])
    rows.append([int(row['ID'])] + list(res))

out = pd.DataFrame(rows, columns=['ID','Value','Delta','Gamma','Vega','Rho','Theta'])
out.to_csv('testout12_2.csv', index=False)
