import pandas as pd
import numpy as np
from scipy.stats import t

input_file = "test7_2.csv"
output_file = "testout8_5.csv"

data = pd.read_csv(input_file)
returns = data['x1']

alpha = 0.95

df, loc, scale = t.fit(returns)

q = t.ppf(1 - alpha, df)

ES = loc - scale * ((df + q**2) / ((df - 1) * (1 - alpha))) * t.pdf(q, df)

result = pd.DataFrame({
    "ES Absolute": [abs(ES)],
    "ES Diff from Mean": [loc - ES]
})

result.to_csv(output_file, index=False)