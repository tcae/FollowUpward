# import sys
from datetime import datetime  # , timedelta
import pandas as pd
import env_config as env
from env_config import Env
# from local_xch import Xch, DATA_KEYS

print("cached_crypto_data init")


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


def save_asset_dataframe(df, base):
    """ saves the base/quote data
    """
    print("{}: writing {} {} tics ({} - {})".format(
        datetime.now().strftime(Env.dt_format), env.sym_of_base(base),
        len(df), df.index[0].strftime(Env.dt_format),
        df.index[len(df)-1].strftime(Env.dt_format)))
    sym = env.sym_of_base(base)
    fname = Env.data_path + sym + "_DataFrame.h5"
    # df.to_msgpack(fname)
    df.to_hdf(fname, sym, mode="w")


def load_asset_dataframe(base, limit=None):
    """ loads the base/quote data
    """
    df = None
    fname = Env.data_path + env.sym_of_base(base) + "_DataFrame.h5"
    sym = env.sym_of_base(base)
    try:
        # df = pd.read_msgpack(fname)
        df = pd.read_hdf(fname, sym)
        print("{}: load {} {} tics ({} - {})".format(
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
    return df


if __name__ == "__main__":
    print("chached crypto data with no actions in __main__")
