# VaR from Simulation
import pandas as pd

INPUT = "test7_2.csv"
OUTPUT = "testout8_3.csv"
df = pd.read_csv(INPUT)

var_abs = df['x1'].quantile(0.5)
var_diff_from_mean = df['x1'].quantile(0.75)

out = pd.DataFrame({
    "VaR Absolute": [var_abs],
    "VaR Diff from Mean": [var_diff_from_mean]
})
out.to_csv(OUTPUT, index=False)
