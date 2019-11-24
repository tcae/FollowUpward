# import sys
from datetime import datetime  # , timedelta
import pandas as pd
import env_config as env
from env_config import Env
# from local_xch import Xch, DATA_KEYS
from local_xch import Xch


""" provides pandas prepresentation of cached crypto data. """


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


def merge_asset_dataframe(path, base):
    """ extends the time range of availble usdt data into the past by mapping base_btc * btc_usdt
    """
    # "loads the object via msgpack"
    fname = path + "btc_usdt" + "_DataFrame.msg"
    btcusdt = load_asset_dataframe("btc")
    if base != "btc":
        fname = path + base + "_btc" + "_DataFrame.msg"
        basebtc = load_asset_dataframefile(fname)
        dfdescribe(f"{base}-btc", basebtc)
        fname = path + base + "_usdt" + "_DataFrame.msg"
        baseusdt = load_asset_dataframefile(fname)
        dfdescribe(f"{base}-usdt", baseusdt)
        if (baseusdt.index[0] <= basebtc.index[0]) or (baseusdt.index[0] <= btcusdt.index[0]):
            basemerged = baseusdt
        else:
            basebtc = basebtc[basebtc.index.isin(btcusdt.index)]
            basemerged = pd.DataFrame(btcusdt)
            basemerged = basemerged[basemerged.index.isin(basebtc.index)]
            for key in Xch.data_keys:
                if key != "volume":
                    basemerged[key] = basebtc[key] * btcusdt[key]
            basemerged["volume"] = basebtc.volume
            dfdescribe(f"{base}-btc-usdt", basemerged)

            baseusdt = baseusdt[baseusdt.index.isin(basemerged.index)]
            assert not baseusdt.empty
            basemerged.loc[baseusdt.index] = baseusdt[:]  # take values of cusdt where available
    else:
        basemerged = btcusdt
    dfdescribe(f"{base}-merged", basemerged)

    save_asset_dataframe(basemerged, Env.data_path, base + "usdt")

    return basemerged
