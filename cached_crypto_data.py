# import sys
from datetime import datetime  # , timedelta
import pandas as pd
import env_config as env
from env_config import Env
from local_xch import Xch


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
    ohlcv = df.loc[:, Xch.data_keys]
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
    drop = [col for col in df.columns if col not in Xch.data_keys]
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

    drop = [col for col in df.columns if col not in Xch.data_keys]
    if len(drop) > 0:
        print(f"load asset df: dropping {drop}")
        df = df.drop(columns=drop)

    if pd.isna(df).any().any():
        print(f"Warning: identified NaN in loaded data - will be filled")
        for key in Xch.data_keys:
            if pd.isna(df[key]).any():
                print(df.loc[df[key][pd.isna(df[key])].index])
        df = df.fillna(method='ffill')
        save_asset_dataframe(df, base, path)
    return df


"""
CryptoData as root class to do most of the pandas handling
Features class wiath
new_data to generate data of timerange last/minutes - this is the heart
    of all new features and targets but also for prediction data
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


class CryptoData:
    """ Abstract superclass for samples set handling.
    """

    def __init__(self):
        self.path = Env.data_path
        try:
            self.sets_split = pd.read_csv(env.sets_split_fname(), skipinitialspace=True, sep="\t")
        except IOError:
            print(f"pd.read_csv({env.sets_split_fname()}) IO error")

    def history(self):
        """ Returns the number of history sample minutes
            excluding the minute under consideration required to calculate a new sample data.
        """
        return 0

    def keys(self):
        "returns the list of element keys"
        return ["missing subclass implementation"]

    def mnemonic(self, base: str):
        "returns a string that represents this class/base as mnemonic, e.g. to use it in file names"
        return "missing subclass implementation"

    def new_data(self, base: str, last: pd.Timestamp, minutes: int):
        """ Downloads or calculates new data for 'minutes' samples up to and including last.
            This is the core method to be implemented by subclasses.
        """
        return None

    def fname(self, base: str):
        "returns a file name to load or store data in h5 format"
        fname = self.path + self.mnemonic(base) + "_df.h5"
        return fname

    def get_data(self, base: str, last: pd.Timestamp, minutes: int):
        """ Loads and downloads/calculates new data for 'minutes' samples up to and including last.
            If minutes == 0 then all saved data is returned and supplemented up to last.
            If minutes == 0 and there is no saved data then None is returned.
        """
        df = self.load_data(base)
        first = last - pd.Timedelta(minutes, unit="T")
        if (df is None) or (last > df.index[-1]):
            diffmin = int((last - df.index[-1]) / pd.Timedelta(1, unit="T"))
            ndf = self.new_data(base, last, diffmin)
            df = pd.concat([df, ndf], join="outer", axis=0, keys=Env.bases)
        df = df.loc[(df.index > first) & (df.index <= last), self.keys()]
        return df

    def load_data(self, base: str):
        """ Loads all saved data and returns it as a data frame.
        """
        try:
            df = pd.read_hdf(self.fname(base), self.mnemonic(base))
            return df[Xch.data_keys]
        except IOError:
            print(f"{env.timestr()} ERROR: cannot load {self.fname(base)}")
            df = None
        return df

    def save_data(self, base: str, df: pd.DataFrame):
        """ Saves complete data that is expected to be in 'df' and overwrites all previous content.
        """
        try:
            fname = self.fname(base)
            df.to_hdf(self.fname(base), self.mnemonic(base), mode="w")
            print(f"saved {fname} {len(df)} samples from {df.index[0]} until {df.index[-1]}")
        except IOError:
            print(f"{env.timestr()} ERROR: cannot save {self.fname(base)}")

    def set_type_data(self, base: str, set_type: str):
        df = self.load_data(base)
        split_df = self.sets_split.loc[self.sets_split.set_type == set_type]
        all = [df.loc[(df.index >= split_df[ix, "start"]) & (df.index <= split_df[ix, "end"])] for ix in len(split_df)]
        set_type_df = pd.concat(all, join="outer", axis=0)
        print(self.sets_split)
        print(split_df)
        return set_type_df


class Features(CryptoData):
    " Abstract superclass of all feature subclasses."
    pass


class Ohlcv(CryptoData):
    """ Class for binance OHLCV samples handling
    """

    def history(self):
        "no history minutes are requires to download OHLCV data"
        return 0

    def keys(self):
        "returns the list of element keys"
        return Xch.data_keys

    def mnemonic(self, base: str):
        "returns a string that represent the PredictionData class as mnemonic, e.g. to use it in file names"
        return base + "_OHLCV"

    def new_data(self, base: str, last: pd.Timestamp, minutes: int):
        """ Predicts all samples and returns the result.
            set_type specific evaluations can be done using the saved prediction data.
        """
        df = Xch.get_ohlcv(base, minutes, last)
        return df[Xch.data_keys]

    # def load_data(self, base: str):
    #     """ Loads all saved ohlcv data and returns it as a data frame.
    #     """
    #     df = load_asset_dataframe(base, path=self.path)
    #     return df[Xch.data_keys]

    # def save_data(self, base: str, df):
    #     """ Saves complete ohlcv data that is expected to be in 'df' and overwrites all previous content.
    #     """
    #     save_asset_dataframe(df, base, path=self.path)


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


def load_asset(bases):
    """ Loads the cached history data as well as live data from the xch
    """

    print(bases)  # Env.usage.bases)
    for base in bases:
        print(f"supplementing {base}")
        hdf = load_asset_dataframe(base, path=Env.data_path)
        # hdf.index.tz_localize(tz='UTC')

        last = (hdf.index[len(hdf)-1])
        # last = (hdf.index[len(hdf)-1]).tz_localize(tz='UTC')
        now = pd.Timestamp.utcnow()
        diffmin = int((now - last)/pd.Timedelta(1, unit='T'))
        ohlcv_df = Xch.get_ohlcv(base, diffmin, now)
        if ohlcv_df is None:
            print("skipping {}".format(base))
            continue
        tix = hdf.index[len(hdf)-1]
        if tix == ohlcv_df.index[0]:
            hdf = hdf.drop([tix])  # the last saved sample is incomple and needs to be updated
            # print("updated last sample of saved cache")
        hdf = pd.concat([hdf, ohlcv_df], sort=False)
        ok2save = check_df(hdf)
        if ok2save:
            save_asset_dataframe(hdf, base, path=Env.data_path)
        else:
            print(f"merged df checked: {ok2save} - dataframe not saved")


if __name__ == "__main__":
    # env.test_mode()
    # load_asset(Env.usage.bases)
    cd = CryptoData()
