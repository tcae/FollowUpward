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


def save_asset_dataframe(df, path, cur_pair):
    """saves the object via msgpack
    """
    # cur_pair = cur_pair.replace("/", "_")
    fname = path + cur_pair + "_DataFrame.msg"
    print("{}: writing {} {} tics ({} - {})".format(
        datetime.now().strftime(Env.dt_format), cur_pair, len(df), df.index[0].strftime(Env.dt_format),
        df.index[len(df)-1].strftime(Env.dt_format)))
    df.to_msgpack(fname)


def load_asset_dataframefile(fname):
    """ loads the object via msgpack
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
    return df


def load_asset_dataframe(base):
    """ loads the object via msgpack
    """
    fname = Env.data_path + base + f"_{Env.quote}" + "_DataFrame.msg"
    dfbu = load_asset_dataframefile(fname)
    if dfbu is None:
        raise env.MissingHistoryData("Cannot load {}".format(fname))
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
