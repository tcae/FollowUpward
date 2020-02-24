# import sys
from datetime import datetime  # , timedelta
import pandas as pd
import env_config as env
from env_config import Env

print("cached_crypto_data init")
data_keys = ["open", "high", "low", "close", "volume"]  # sequence is important, don't make it a set!


""" provides pandas prepresentation of cached crypto data.
    The module shall abstract from the actual features, i.e. provide
    service functions and a base class that can be subclasses for
    different types of features.
"""


def dfdescribe(desc, df):
    print(desc)
    print(df.describe())
    print(df.head())
    print(df.tail())


def only_ohlcv(df):
    """ Drops all columns but the ohlcv columns and returns the resulting DataFrame
    """
    ohlcv = df.loc[:, data_keys]
    return ohlcv


def save_asset_dataframe(df, base, path="missing path"):
    """ saves the base/quote data
    """
    print("{}: writing ohlcv of {} {} tics ({} - {})".format(
        datetime.now().strftime(Env.dt_format), env.sym_of_base(base),
        len(df), df.index[0].strftime(Env.dt_format),
        df.index[len(df)-1].strftime(Env.dt_format)))
    sym = env.sym_of_base(base)
    fname = path + sym + "_DataFrame.h5"
    drop = [col for col in df.columns if col not in data_keys]
    if len(drop) > 0:
        print(f"save asset df: dropping {drop}")
        df = df.drop(columns=drop)

    # df.to_msgpack(fname)
    df.to_hdf(fname, sym, mode="w")


def load_asset_dataframe(base, path="missing path", limit=None):
    """ loads the base/quote data
    """
    df = None
    fname = path + env.sym_of_base(base) + "_DataFrame.h5"
    sym = env.sym_of_base(base)
    try:
        # df = pd.read_msgpack(fname)
        df = pd.read_hdf(fname, sym)
        print("{}: loaded {} {} tics ({} - {})".format(
            datetime.now().strftime(Env.dt_format), fname, len(df), df.index[0].strftime(Env.dt_format),
            df.index[len(df)-1].strftime(Env.dt_format)))
    except IOError:
        print(f"{env.timestr()} load_asset_dataframefile ERROR: cannot load {fname}")
    except ValueError:
        return None
    if df is None:
        raise env.MissingHistoryData("Cannot load {}".format(fname))
    if (limit is not None) and (limit < len(df)):
        df = df.drop(df.index[:len(df)-limit])  # only hold the last `limit` minutes

    drop = [col for col in df.columns if col not in data_keys]
    if len(drop) > 0:
        print(f"load asset df: dropping {drop}")
        df = df.drop(columns=drop)

    if pd.isna(df).any().any():
        print(f"Warning: identified NaN in loaded data - will be filled")
        for key in data_keys:
            if pd.isna(df[key]).any():
                print(df.loc[df[key][pd.isna(df[key])].index])
        df = df.fillna(method='ffill')
        save_asset_dataframe(df, base, path)
    return df

"""
CryptoData as root class to do most of the pandas handling
Features class wiath
new_data to generate data of timerange last/minutes - this is the heart of all new features and targets but also for prediction data
supplement_data to load cache and add data up to last with help of new data
get_data gets last?minutes data which is either loaded or generated
menmonic for fname construction
fname for same save/load fname
save_data saves all data
load data loads all data
get_set_type to get labelled configed data
history to denote number of history samples used for new_data
keys == data columns
target_dict = target categories if Target subclass
"""

if __name__ == "__main__":
    print("chached crypto data with no actions in __main__")
