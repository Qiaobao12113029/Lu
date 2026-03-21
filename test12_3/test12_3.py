import numpy as np
import pandas as pd

def american_discrete_div(option_type, S, K, T, r, sigma, div_times, div_amts, N=400):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    ex_steps = [(int(td / dt), D) for td, D in zip(div_times, div_amts)]

    j = np.arange(N + 1)
    ST = S * u**j * d**(N - j)
    for step, D in ex_steps:
        ST -= D
    V = np.maximum(ST - K, 0) if option_type.lower() == 'call' else np.maximum(K - ST, 0)

    for i in range(N - 1, -1, -1):
        ST_i = S * u**np.arange(i + 1) * d**(i - np.arange(i + 1))
        for step, D in ex_steps:
            if step <= i:
                ST_i -= D
        V = disc * (p * V[1:i+2] + (1 - p) * V[0:i+1])
        if option_type.lower() == 'call':
            V = np.maximum(V, ST_i - K)
        else:
            V = np.maximum(V, K - ST_i)

    return V[0]

df = pd.read_csv('test12_3.csv')
rows = []
for _, row in df.iterrows():
    T = row['DaysToMaturity'] / row['DayPerYear']
    div_times = [int(d) / row['DayPerYear'] for d in str(row['DividendDates']).split(',')]
    div_amts  = [float(d) for d in str(row['DividendAmts']).split(',')]

    value = american_discrete_div(
        row['Option Type'], row['Underlying'], row['Strike'], T,
        row['RiskFreeRate'], row['ImpliedVol'], div_times, div_amts
    )
    rows.append([row['ID'], value])

out = pd.DataFrame(rows, columns=['ID', 'Value'])
out.to_csv('testout12_3.csv', index=False)
