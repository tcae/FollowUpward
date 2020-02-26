# from enum import Enum
import sys
import platform
from datetime import datetime  # , timedelta
import pandas as pd

PICKLE_EXT = ".pydata"  # pickle file extension


def nowstr():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


class MissingHistoryData(Exception):
    pass


class Env():
    quote = "usdt"
    sym_sep = "_"  # symbol seperator
    xch_sym_sep = "/"  # ccxt symbol seperator
    smaller_16gb_ram = True
    dt_format = "%Y-%m-%d_%Hh%Mm"
    calc = None
    usage = None

    def __init__(self, calc, usage):
        print(f"{type(calc)}/{type(usage)}")
        Env.data_path = calc.data_path_prefix + usage.data_path_suffix
        Env.conf_fname = usage.conf_fname
        Env.test_mode = usage.test_mode
        Env.cache_path = f"{calc.data_path_prefix}cache/"
        Env.bases = usage.bases
        Env.model_path = f"{calc.other_path_prefix}classifier/"
        Env.tfblog_path = f"{calc.other_path_prefix}tensorflowlog/"
        Env.auth_file = calc.auth_path_prefix + "auth_Tst1.json"
        Env.calc = calc
        Env.usage = usage

    def tensorboardpath():
        path = f"{Env.tfblog_path}{timestr()}"
        # was before f"{TFBLOG_PATH}{timestr()}talos{self.talos_iter}-"
        return path


class Calc():
    home = ""
    data_path_prefix = ""
    other_path_prefix = ""


class Ubuntu(Calc):
    home = "/home/tor/"
    data_path_prefix = home + "crypto/"
    other_path_prefix = home + "crypto/"
    auth_path_prefix = home + ".catalyst/data/exchanges/binance/"


class Osx(Calc):
    home = "/Users/tc/"
    data_path_prefix = home + "crypto/"
    other_path_prefix = home + "crypto/"
    auth_path_prefix = home + ".catalyst/data/exchanges/binance/"


class Colab(Calc):
    home = "/content/gdrive/My Drive/"
    data_path_prefix = home
    other_path_prefix = home
    auth_path_prefix = home


class Floydhub(Calc):
    home = ""
    data_path_prefix = "/floyd/input/"
    other_path_prefix = home
    auth_path_prefix = home


class Usage():
    data_path = "data path not initialized"
    bases = []
    conf_fname = "conf fname not initialized"
    test_mode = False


class Production(Usage):
    data_path_suffix = "Features/"
    bases = ["btc", "xrp", "eos", "bnb", "eth", "neo", "ltc", "trx"]
    conf_fname = "target_5_sets_split.config"
    set_split_fname = "adapt_sets_split.csv"
    test_mode = False


class Test(Usage):
    data_path_suffix = "TestFeatures/"
    bases = ["xrp"]
    # conf_fname = "target_5_sets_split_unit-test.config"
    conf_fname = "target_5_sets_split.config"
    set_split_fname = "adapt_sets_split.csv"
    test_mode = True


def config_ok():
    return len(Env.data_path) > 0


def sets_config_fname():
    cfname = Env.data_path + Env.conf_fname
    return cfname


def sets_split_fname():
    cfname = Env.data_path + Env.usage.set_split_fname
    return cfname


def timestr(ts=None):
    if ts is None:
        return datetime.now().strftime(Env.dt_format)
    else:
        return pd.to_datetime(ts).strftime(Env.dt_format)


def time_in_index(dataframe_with_timeseriesindex, tic):
    return True in dataframe_with_timeseriesindex.index.isin([tic])


def sym_of_base(base):
    s = base.lower() + Env.sym_sep + Env.quote.lower()
    return s


def base_and_quote(symbol):
    """ returns base and quote as lowercase tuple of a given symbol
    """
    symbol = symbol.lower()
    symbol = symbol.replace(Env.xch_sym_sep, Env.sym_sep)  # if required convert ccxt separtor
    slist = symbol.split(Env.sym_sep)
    assert(len(slist) == 2)
    base = slist[0]
    quote = slist[1]
    return (base, quote)


def base_of_sym(sym):
    s = sym.lower()
    q = Env.quote.lower()
    ix = s.find(q)
    if ix < 0:
        raise ValueError(f"base_of_sym {sym}: no quote {Env.quote} found")
    if ix == 0:
        raise ValueError(f"base_of_sym {sym}: no base found")
    if not s[ix-1].isalpha():  # seperation character found
        ix -= 1
    b = s[0:ix]
    return b


class Tee(object):

    def __init__(self):
        name = f"{Env.model_path}Log_{timestr()}.txt"  # f"{env.MODEL_PATH}Log_{env.timestr()}.txt"
        self.file = open(name, "w")
        self.stdout = sys.stdout
        sys.stdout = self

    def close(self):
        if self.stdout is not None:
            sys.stdout = self.stdout
            self.stdout = None
        if self.file is not None:
            self.file.close()
            self.file = None

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __del__(self):
        self.close()


def test_mode():
    if platform.node() == "iMac.local":
        Env(Osx(), Test())
    elif platform.node() == "tor-XPS-13-9380":
        Env(Ubuntu(), Test())


def check_df(df):
    # print(df.head())
    diff = pd.Timedelta(value=1, unit="T")
    last = this = df.index[0]
    ok = True
    ix = 0
    for tix in df.index:
        if this != tix:
            print(f"ix: {ix} last: {last} tix: {tix} this: {this}")
            ok = False
            this = tix
        last = tix
        this += diff
        ix += 1
    return ok


# TODO the following is a Stackoverflow monkey patch to catch warnings. to be beatified
old_f = sys.stderr


class F:
    def write(self, x):
        if "A value is trying to be set on a copy of a slice from a DataFrame." in x:
            print("To be investigated")
        old_f.write(x)

    def flush(self):
        old_f.flush()


sys.stderr = F()

if platform.node() == "iMac.local":
    Env(Osx(), Production())
elif platform.node() == "tor-XPS-13-9380":
    Env(Ubuntu(), Production())
