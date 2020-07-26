# from enum import Enum
import logging
import sys
import platform
from datetime import datetime  # , timedelta
import pandas as pd
# import tensorflow as tf


logger = logging.getLogger(__name__)

PICKLE_EXT = ".pydata"  # pickle file extension


def nowstr():
    # return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return pd.Timestamp.now(tz=Env.tz).to_pydatetime().strftime("%Y-%m-%d_%H:%M:%S")


class MissingHistoryData(Exception):
    pass


class Env():
    quote = "usdt"
    tz = "Europe/Amsterdam"
    sym_sep = "_"  # symbol seperator
    xch_sym_sep = "/"  # ccxt symbol seperator
    smaller_16gb_ram = True
    dt_format = "%Y-%m-%d_%Hh%Mm"
    calc = None
    usage = None

    def __init__(self, calc, usage):
        # logger.info(f"{type(calc)}/{type(usage)}")
        # first initialize environment then configure logging and then start logging
        Env.data_path = calc.data_path_prefix + usage.data_path_suffix
        Env.test_mode = usage.test_mode
        Env.cache_path = f"{calc.data_path_prefix}cache/"
        Env.bases = usage.bases
        Env.training_bases = usage.training_bases
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
    bases = [
        "btc", "xrp", "eos", "bnb", "eth", "neo", "ltc", "trx", "zrx", "bch",
        "etc", "link", "ada", "matic", "xtz", "zil", "omg", "xlm", "zec",
        "tfuel", "theta", "ont", "vet", "iost", "sinus", "triangle"]
    training_bases = [
        "btc", "xrp", "eos", "bnb", "eth", "neo", "ltc", "trx"]
    set_split_fname = "sets_split.csv"
    test_mode = False


class Test(Usage):
    data_path_suffix = "TestFeatures/"
    training_bases = ["sinus"]
    bases = ["sinus"]
    set_split_fname = "sets_split.csv"
    test_mode = True


def config_ok():
    return len(Env.data_path) > 0


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


class Tee_outdated(object):
    file = None
    stdout = None
    stderr = None
    special_log = None

    def __init__(self):
        self.reset_path()

    @classmethod
    def reset_path(cls):
        cls.set_path(Env.model_path, base_log=True)

    @classmethod
    def set_path(cls, log_path: str, base_log=False):
        name = f"{log_path}Log_{timestr()}.txt"
        try:
            new_file = open(name, "w")
            if base_log:
                if cls.file is not None:
                    cls.file.close()
                cls.file = new_file
            else:
                if cls.special_log is not None:
                    cls.special_log.close()
                cls.special_log = new_file
            if cls.stdout is None:
                cls.stdout = sys.stdout
                sys.stdout = cls
            if cls.stderr is None:
                cls.stderr = sys.stderr
                sys.stderr = cls
        except IOError:
            print(f"{timestr()} ERROR: cannot redirect stdout and stderr to {name}")

    @classmethod
    def close(cls):
        if cls.stdout is not None:
            sys.stdout = cls.stdout
            cls.stdout = None
        if cls.stderr is not None:
            sys.stderr = cls.stderr
            cls.stderr = None
        if cls.file is not None:
            cls.file.close()
            cls.file = None
        if cls.special_log is not None:
            cls.special_log.close()
            cls.special_log = None

    @classmethod
    def write(cls, data):
        if cls.file is not None:
            cls.file.write(data)
        if cls.special_log is not None:
            cls.special_log.write(data)
        if cls.stdout is not None:
            cls.stdout.write(data)
        if cls.stderr is not None:
            cls.stderr.write(data)

    @classmethod
    def flush(cls):
        if cls.file is not None:
            cls.file.flush()
        if cls.special_log is not None:
            cls.special_log.flush()
        if cls.stdout is not None:
            cls.stdout.flush()
        if cls.stderr is not None:
            cls.stderr.flush()

    @classmethod
    def __del__(cls):
        cls.close()


def test_mode():
    if platform.node() == "iMac.local":
        Env(Osx(), Test())
    elif platform.node() == "tor-XPS-13-9380":
        Env(Ubuntu(), Test())
    logger.info(">>> test mode <<<")


def check_df(df):
    # logger.debug(str(df.head()))
    diff = pd.Timedelta(value=1, unit="T")
    last = this = df.index[0]
    ok = True
    ix = 0
    for tix in df.index:
        if this != tix:
            logger.debug(f"ix: {ix} last: {last} tix: {tix} this: {this}")
            ok = False
            this = tix
        last = tix
        this += diff
        ix += 1
    return ok


class Tee(object):
    logger = None  # will be root logger
    formatter = None
    streamhandler = None
    filehandler = None
    filehandler2 = None
    stderr = None

    def __init__(self, log_prefix="Log"):
        self.init_tee(log_prefix)

    @classmethod
    def init_tee(cls, log_prefix="Log"):
        # cls.stderr = sys.stderr
        # sys.stderr = Tee  # intercept stderr messages in write method

        fname = f"{Env.model_path}{log_prefix}_{timestr()}.csv"
        cls.logger = logging.getLogger()
        cls.formatter = logging.Formatter(
            fmt="%(asctime)s \t%(levelname)s \t%(name)s \t%(funcName)s: \t%(message)s",
            datefmt="%Y-%m-%d_%H:%M:%S")
        if cls.streamhandler is not None:
            cls.logger.removeHandler(cls.streamhandler)
        cls.streamhandler = logging.StreamHandler(sys.stderr)
        cls.streamhandler.setFormatter(cls.formatter)
        if cls.filehandler is not None:
            cls.logger.removeHandler(cls.filehandler)
        cls.filehandler = logging.FileHandler(filename=fname, mode="w")
        cls.filehandler.setFormatter(cls.formatter)
        cls.logger.addHandler(cls.streamhandler)
        cls.logger.addHandler(cls.filehandler)
        cls.logger.setLevel(level=logging.DEBUG)
        if Env.test_mode:
            cls.logger.info("test mode")
        else:
            cls.logger.info("production mode")
        tf_logger = logging.getLogger("tensorflow")  # tf.get_logger()
        if tf_logger is not None:
            tf_logger.addHandler(cls.filehandler)
        else:
            logger.info("no tensorflow logger")
        ccxt_logger = logging.getLogger("ccxt")
        if ccxt_logger is not None:
            ccxt_logger.setLevel(level=logging.INFO)
        else:
            logger.info("no ccxt logger")
        urllib3_logger = logging.getLogger("urllib3")
        if urllib3_logger is not None:
            urllib3_logger.setLevel(level=logging.WARNING)
        else:
            logger.info("no urllib3 logger")
        numexpr_logger = logging.getLogger("numexpr")
        if numexpr_logger is not None:
            numexpr_logger.setLevel(level=logging.WARNING)
        else:
            logger.info("no numexpr logger")

    @classmethod
    def reset_path(cls):
        if cls.filehandler2 is not None:
            cls.logger.removeHandler(cls.filehandler2)
            cls.filehandler2 = None

    @classmethod
    def set_path(cls, log_path: str, log_prefix="Log"):
        fname = f"{log_path}{log_prefix}_{timestr()}.csv"
        if cls.filehandler2 is not None:
            cls.logger.removeHandler(cls.filehandler2)
        cls.filehandler2 = logging.FileHandler(filename=fname, mode="w")
        if cls.filehandler2 is not None:
            cls.filehandler2.setFormatter(cls.formatter)
            cls.logger.addHandler(cls.filehandler2)

    @classmethod
    def write(cls, x):
        intercept_str = "Passing (type, 1) or '1type' as a synonym of type is deprecated"
        # intercept_str = "A value is trying to be set on a copy of a slice from a DataFrame."
        if intercept_str in x:
            print(f"Detected intercept str: {intercept_str}")
        cls.stderr.write(x)

    @classmethod
    def flush(cls):
        cls.stderr.flush()

    @classmethod
    def close(cls):  # only present for backward compatibility
        pass


if platform.node() == "iMac.local":
    Env(Osx(), Production())
elif platform.node() == "tor-XPS-13-9380":
    Env(Ubuntu(), Production())
Tee()
