# import sys
from datetime import datetime  # , timedelta
import pandas as pd
import env_config as env


""" provides pandas prepresentation of cached crypto data. """


def save_asset_dataframe(df, path, cur_pair):
    # "saves the object via msgpack"
    # cur_pair = cur_pair.replace("/", "_")
    fname = path + cur_pair + "_DataFrame.msg"
    print("{}: writing {} {} tics ({} - {})".format(
        datetime.now().strftime(env.DT_FORMAT), cur_pair, len(df), df.index[0].strftime(env.DT_FORMAT),
        df.index[len(df)-1].strftime(env.DT_FORMAT)))
    df.to_msgpack(fname)


def load_asset_dataframefile(fname):
    # "loads the object via msgpack"
    df = None
    try:
        df = pd.read_msgpack(fname)
        print("{}: load {} {} tics ({} - {})".format(
            datetime.now().strftime(env.DT_FORMAT), fname, len(df), df.index[0].strftime(env.DT_FORMAT),
            df.index[len(df)-1].strftime(env.DT_FORMAT)))
    except IOError:
        print(f"{env.EnvCfg.timestr()} load_asset_dataframefile ERROR: cannot load {fname}")
    except ValueError:
        return None
    return df


def load_asset_dataframe(path, base):
    # "loads the object via msgpack"
    fname = path + base + f"_{env.QUOTE}" + "_DataFrame.msg"
    dfbu = load_asset_dataframefile(fname)
    if dfbu is None:
        raise env.MissingHistoryData("Cannot load {}".format(fname))
    return dfbu
