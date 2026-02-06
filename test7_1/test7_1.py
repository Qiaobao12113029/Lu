import pandas as pd
import numpy as np
data = pd.read_csv("test7_1.csv", header=None)
values = pd.to_numeric(data.stack(), errors="coerce").dropna().values
mu = np.mean(values)
sigma = np.std(values, ddof=0)

output_df = pd.DataFrame({
    "mu": [mu],
    "sigma": [sigma]
})

output_df.to_csv("testout7_1.csv", index=False)
