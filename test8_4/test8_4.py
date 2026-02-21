import pandas as pd
import numpy as np
from scipy.stats import norm

input_file = "test7_1.csv"
output_file = "testout8_4.csv"

data = pd.read_csv(input_file)
returns = data['x1']
mu = returns.mean()
sigma = returns.std(ddof=1)
alpha = 0.95
z = norm.ppf(1 - alpha)
ES = mu - sigma * (norm.pdf(z) / (1 - alpha))
result = pd.DataFrame({
    "ES Absolute": [abs(ES)],
    "ES Diff from Mean": [mu - ES]
})

result.to_csv(output_file, index=False)
