import pandas as pd
df = pd.read_csv("test6.csv", index_col=0)
ret = df.pct_change()
ret = ret.iloc[1:]
ret.reset_index().to_csv("testout6_1.csv", index=False)

