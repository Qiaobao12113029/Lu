import pandas as pd
import numpy as np
from scipy.stats import t

data = pd.read_csv("test7_2.csv", header=None)

values = pd.to_numeric(data.stack(), errors="coerce").dropna().values

nu, mu, sigma = t.fit(values)   

output_df = pd.DataFrame({
    "mu": [mu],
    "sigma": [sigma],
    "nu": [nu]
})

output_df.to_csv("testout7_2.csv", index=False)

