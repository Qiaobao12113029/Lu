import pandas as pd
import numpy as np

df = pd.read_csv("test6.csv", index_col=0)
log_ret = np.log(df / df.shift(1)).iloc[1:]
log_ret.reset_index().to_csv("testout6_2.csv",index=False)
