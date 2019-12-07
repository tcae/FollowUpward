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


def asset_fname(base):
    fname = Env.data_path + env.sym_of_base(base) + "_DataFrame.msg"
    return fname


def save_asset_dataframefile(df, fname):
    """saves the file fname data via msgpack
    """
    df.to_msgpack(fname)


def save_asset_dataframe(df, base):
    """ saves the base/quote data via msgpack
    """
    print("{}: writing {} {} tics ({} - {})".format(
        datetime.now().strftime(Env.dt_format), env.sym_of_base(base), len(df), df.index[0].strftime(Env.dt_format),
        df.index[len(df)-1].strftime(Env.dt_format)))
    save_asset_dataframefile(df, asset_fname(base))


def load_asset_dataframefile(fname):
    """ loads the file fname data via msgpack
    """
    df = None
    try:
        df = pd.read_msgpack(fname)
        print("{}: load {} {} tics ({} - {})".format(
            datetime.now().strftime(Env.dt_format), fname, len(df), df.index[0].strftime(Env.dt_format),
            df.index[len(df)-1].strftime(Env.dt_format)))
    except IOError:
        print(f"{env.EnvCfg.timestr()} load_asset_dataframefile ERROR: cannot load {fname}")
    except ValueError:
        return None
    if df is None:
        raise env.MissingHistoryData("Cannot load {}".format(fname))
    return df


def load_asset_dataframe(base):
    """ loads the base/quote data via msgpack
    """
    dfbu = load_asset_dataframefile(asset_fname(base))
    return dfbu


def load_cache_dataframe(base):
    """ loads the object via msgpack
    """
    fname = Env.cache_path + base + f"_{Env.quote}" + "_DataFrame.msg"
    dfbu = load_asset_dataframefile(fname)
    if dfbu is None:
        raise env.MissingHistoryData("Cannot load {}".format(fname))
    return dfbu


if __name__ == "__main__":
    print("chached crypto data with no actions in __main__")
