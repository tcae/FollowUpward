import pandas as pd
import os
import platform


def df_drop():
    df = pd.DataFrame(
        index=pd.date_range(
            end=pd.Timestamp(year=2019, month=12, day=22, hour=23, minute=50),
            freq="T", periods=20), columns=["A", "B", "C"],
        data=[[m+r for m in range(3)] for r in range(20)])
    print(df)
    limit = 4
    ddf = df.drop(df.index[:len(df)-limit-1])
    print(ddf)
    cdf = ddf.append(df)
    print(cdf)


print([[m+r for m in range(3)] for r in range(20)])
df_drop()

print(os.name)
print(platform.system())
print(platform.node())
