import pandas as pd
import hvplot.pandas
import crypto_targets_features as ctf

df = pd.read_msgpack("df.msg")
df = df.head(1000000)
df = df.loc[:,["close","high"]]
print(df.head())
# df.hvplot()
df.hvplot(datashade=True)
