from enum import Enum
from datetime import datetime  # , timedelta
import pandas as pd

DATA_PATH = ""
OTHER_PATH_PREFIX = ""
DATA_PATH_PREFIX = ""
HOME = ""
BASES = []
TIME_AGGS = {}
QUOTE = "usdt"
SMALLER_16GB_RAM = True
DT_FORMAT = "%Y-%m-%d_%Hh%Mm"


def nowstr():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


class MissingHistoryData(Exception):
    pass


class Calc(Enum):
    ubuntu = 1
    osx = 2
    floydhub = 3
    colab = 4


class Usage(Enum):
    test = 11
    production = 12


def set_environment(test_conf, this_env):
    global HOME
    global DATA_PATH
    global OTHER_PATH_PREFIX
    global DATA_PATH_PREFIX
    global BASES
    global TIME_AGGS
    global CACHE_PATH
    global AUTH_FILE
    global MODEL_PATH
    global TFBLOG_PATH

    if this_env == Calc.osx:
        HOME = "/Users/tc/"
        DATA_PATH_PREFIX = HOME + "crypto/"
        OTHER_PATH_PREFIX = HOME + "crypto/"
    elif this_env == Calc.ubuntu:
        HOME = "/home/tor/"
        DATA_PATH_PREFIX = HOME + "crypto/"
        OTHER_PATH_PREFIX = HOME + "crypto/"
    elif this_env == Calc.colab:
        HOME = "/content/gdrive/My Drive/"
        DATA_PATH_PREFIX = HOME
        OTHER_PATH_PREFIX = HOME
        assert(not test_conf)
    elif this_env == Calc.floydhub:
        HOME = ""
        DATA_PATH_PREFIX = "/floyd/input/"
        OTHER_PATH_PREFIX = HOME
        assert(not test_conf)
    else:
        raise ValueError(f"configuration fault in this_env: {this_env}")

    if test_conf == Usage.test:
        DATA_PATH = f"{DATA_PATH_PREFIX}TestFeatures/"  # local execution
        TIME_AGGS = {1: 10, 5: 10, 15: 10, 60: 10, 4*60: 10, 24*60: 10}
        # BASES = ["xrp", "bnb", "eos"]
        # BASES = ["xrp", "eos"]
        BASES = ["xrp"]
    elif test_conf == Usage.production:
        DATA_PATH = f"{DATA_PATH_PREFIX}Features/"
        BASES = ["xrp", "eos", "bnb", "btc", "eth", "neo", "ltc", "trx"]
        TIME_AGGS = {1: 10, 5: 10, 15: 10, 60: 10, 4*60: 10, 24*60: 10}
    else:
        raise ValueError(f"configuration fault in test_conf: {test_conf}")

    MODEL_PATH = f"{OTHER_PATH_PREFIX}classifier/"
    TFBLOG_PATH = f"{OTHER_PATH_PREFIX}tensorflowlog/"
    CACHE_PATH = f"{DATA_PATH_PREFIX}cache/"
    AUTH_FILE = HOME + ".catalyst/data/exchanges/binance/auth.json"


def config_ok():
    return len(DATA_PATH) > 0


def sets_config_fname():
    cfname = DATA_PATH + "target_5_sets_split.config"
    return cfname


def timestr(ts=None):
    if ts is None:
        return datetime.now().strftime(DT_FORMAT)
    else:
        return pd.to_datetime(ts).strftime(DT_FORMAT)


def time_in_index(dataframe_with_timeseriesindex, tic):
    return True in dataframe_with_timeseriesindex.index.isin([tic])


def sym_of_base(base):
    s = f"{base.lower()}_{QUOTE}"
    return s


def base_of_sym(sym):
    s = sym.lower()
    q = QUOTE.lower()
    ix = s.find(q)
    if ix < 0:
        raise ValueError(f"base_of_sym {sym}: no quote {QUOTE} found")
    if ix == 0:
        raise ValueError(f"base_of_sym {sym}: no base found")
    if not s[ix-1].isalpha():  # seperation character found
        ix -= 1
    b = s[0:ix]
    return b
