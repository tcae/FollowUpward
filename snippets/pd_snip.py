""" pandas_snippets to understand pandas with examples
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def df_show(descr: str = "", df: pd.DataFrame = None):
    print(f"{descr} \n{df} \n")


def create_test_df(descr: str, start: str, minutes: int, columns: list):
    """ creates dataframe with minutes index
    """
    df = pd.DataFrame(
        data=np.random.rand(minutes, len(columns)),
        columns=columns, index=pd.date_range(start, periods=minutes, freq="T", tz="Europe/Amsterdam"))
    df_show(descr, df)
    return df


def identify_missing_data(df: pd.DataFrame):
    idx = pd.date_range(start=df.index[0], end=df.index[-1], freq="T", tz="Europe/Amsterdam")
    missing_dates = idx[~idx.isin(df.index)]
    df_show("missing index entries", missing_dates)


def identify_missing_data_ranges(df: pd.DataFrame):
    df = df.reset_index()
    df["gap_last"] = df["index"] - pd.Timedelta(1, unit='T')
    df["gap_first"] = df["index"].shift(1) + pd.Timedelta(1, unit='T')
    df["gap_length"] = df["index"] - df["index"].shift(1)
    df_show("df reset index", df)
    # df["start"] = df.loc[((df["index"].shift(-1) - df["index"])/pd.Timedelta(1, unit='T') > 1), "next"]
    dfg = df.loc[((df["index"] - df["index"].shift(1))/pd.Timedelta(1, unit='T') > 1),
                 ["gap_first", "gap_last", "gap_length"]]
    df_show("df with gaps", dfg)


def fill_missing_data(df: pd.DataFrame):
    idx = pd.date_range(start=df.index[0], end=df.index[-1], freq="T", tz="Europe/Amsterdam")
    df = df.reindex(index=idx, method='ffill')
    df_show("df concat filled", df)


def missing_data():
    """ problem ohlcv load has gaps. pandas has fill methods that can be used
        but is the Datetimeindex in minute frequency properly filled as well?
    """
    cols = ["A", "B", "C"]
    dfa = create_test_df("df1", "2012-10-08 18:15:05", 3, cols)
    dfb = create_test_df("df2", "2012-10-08 18:25:05", 3, cols)
    dfc = pd.concat([dfa, dfb], join="outer")
    df_show("df concat", dfc)

    identify_missing_data(dfc)
    identify_missing_data_ranges(dfc)

    fill_missing_data(dfc)


if __name__ == "__main__":
    missing_data()
