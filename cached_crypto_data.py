# import sys
import logging
# from datetime import datetime  # , timedelta
import timeit
import pandas as pd
import env_config as env
from env_config import Env
from local_xch import Xch


""" provides pandas prepresentation of cached crypto data.
    The module shall abstract from the actual features, i.e. provide
    service functions and a base class that can be subclasses for
    different types of features.
"""

logger = logging.getLogger(__name__)


def show_verbose(df, verbose=True, lines=5):
    if verbose:
        logger.debug(f"entries: {len(df)}")
        if len(df) <= (2*lines):
            logger.debug(f"{df}\n")
        else:
            logger.debug(f"{df.head(lines)} \n{df.tail(lines)} \n")


def dfdescribe(desc, df):
    logger.debug(str(desc))
    logger.debug(str(df.describe(percentiles=[], include='all')))
    show_verbose(verbose=True, lines=2)
    if no_index_gaps(df):
        logger.debug("no index gaps")


def common_timerange(df_list):
    """ limits the datetimerange of the list of given data frames to the common range.
        Returns a data frame list in the same sequence with limited ranges.
    """
    first = max([df.index[0] for df in df_list])
    last = min([df.index[-1] for df in df_list])
    return [df.loc[(df.index >= first) & (df.index <= last)] for df in df_list]


def no_index_gaps(df: pd.DataFrame):
    ix_check = df.index.to_series(keep_tz=True).diff().dt.seconds / 60
    ix_gaps = ix_check.loc[ix_check > 1]  # reduce time differences to those > 60 sec
    if not ix_gaps.empty:
        logger.warning("found index gaps")
        logger.warning(str(ix_gaps))
    # logger.debug(f"index check len: {len(ix_check)}, index check empty: {ix_check.empty}")
    return ix_gaps.empty


def only_ohlcv(df):
    """ Drops all columns but the ohlcv columns and returns the resulting DataFrame
    """
    ohlcv = df.loc[:, Xch.data_keys]
    return ohlcv


def save_asset_dataframe(df, base, path="missing path"):
    """ saves the base/quote data
    """
    logger.info("writing ohlcv of {} {} tics ({} - {})".format(
        env.sym_of_base(base),
        len(df), df.index[0].strftime(Env.dt_format),
        df.index[len(df)-1].strftime(Env.dt_format)))
    sym = env.sym_of_base(base)
    fname = path + sym + "_DataFrame.h5"
    drop = [col for col in df.columns if col not in Xch.data_keys]
    if len(drop) > 0:
        logger.debug(f"save asset df: dropping {drop}")
        df = df.drop(columns=drop)

    df.to_hdf(fname, sym, mode="w")


def load_asset_dataframe(base, path="missing path", limit=None):
    """ loads the base/quote data
    """
    df = None
    fname = path + env.sym_of_base(base) + "_DataFrame.h5"
    sym = env.sym_of_base(base)
    try:
        df = pd.read_hdf(fname, sym)
        logger.info("loaded {} {} tics ({} - {})".format(
            fname, len(df), df.index[0].strftime(Env.dt_format),
            df.index[len(df)-1].strftime(Env.dt_format)))
    except IOError:
        logger.error(f"load_asset_dataframefile ERROR: cannot load {fname}")
    except ValueError:
        return None
    if df is None:
        raise env.MissingHistoryData("Cannot load {}".format(fname))
    if (limit is not None) and (limit < len(df)):
        df = df.drop(df.index[:len(df)-limit])  # only hold the last `limit` minutes

    drop = [col for col in df.columns if col not in Xch.data_keys]
    if len(drop) > 0:
        logger.debug(f"load asset df: dropping {drop}")
        df = df.drop(columns=drop)

    if pd.isna(df).any().any():
        logger.debug(f"Warning: identified NaN in loaded data - will be filled")
        for key in Xch.data_keys:
            if pd.isna(df[key]).any():
                logger.debug(str(df.loc[df[key][pd.isna(df[key])].index]))
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
        self.sets_split = None
        self.missing_file_warning = True

    def history(self):
        """ Returns the number of history sample minutes
            excluding the minute under consideration required to calculate a new sample data.
        """
        return 0

    def keys(self):
        "returns the list of element keys"
        return ["missing subclass implementation"]

    def mnemonic(self):
        "returns a string that represents this class as mnemonic, e.g. to use it in file names"
        return "missing subclass implementation"

    def new_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        """ Downloads or calculates new data from 'first' sample up to and including 'last'.
            This is the core method to be implemented by subclasses.
            If use_cache == False then no saved data is used.
        """
        return None

    def fname(self, base: str):
        "returns a file name to load or store data in h5 format"
        fname = self.path + base + "_" + self.mnemonic() + "_df.h5"
        return fname

    def check_timerange(self, df: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp):
        """ Returns a data frame that is limited to the given timerange.
            Issues a warning if the timerange is smaller than requested.
        """
        assert first is not None
        assert last is not None
        assert first <= last
        df = df.loc[(df.index >= first) & (df.index <= last)]
        if (df.index[0] > first) or (df.index[-1] < last):
            logger.warning(f"index time gap of: {df.index[0] - first} at start, {last - df.index[-1]} at end")
        return df

    def get_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        """ Loads and downloads/calculates new data from 'first' sample up to and including 'last'.
            If use_cache == False then no saved data is used.
        """
        assert first is not None
        assert last is not None
        assert first <= last
        if use_cache:
            df = self.load_data(base)
        else:
            df = None
        if (df is None) or df.empty:
            df = self.new_data(base, first, last, use_cache)
            if (df is None) or df.empty:
                return None
        elif (last > df.index[-1]):
            if first > df.index[-1]:
                df = self.new_data(base, first, last, use_cache)
            else:
                # include df.index[-1] as new data first due to incomplete last minute effect
                ndf = self.new_data(base, df.index[-1], last, use_cache)
                if (ndf is not None) and (not ndf.empty):
                    df = df.loc[df.index < ndf.index[0]]
                    # thereby cutting any time overlaps, e.g. +1 minute due to incomplete last minute effect
                    df = pd.concat([df, ndf], join="outer", axis=0, sort=True, verify_integrity=True)
        df = df.loc[(df.index >= first) & (df.index <= last), self.keys()]
        assert no_index_gaps(df)
        return df

    def load_data(self, base: str):
        """ Loads all saved data and returns it as a data frame.
        """
        try:
            df = pd.read_hdf(self.fname(base), base + "_" + self.mnemonic())
            if isinstance(df, pd.Series):
                df = df.to_frame()
            df = df.loc[:, self.keys()]
        except IOError:
            if self.missing_file_warning:
                logger.warning(f"cannot load {self.fname(base)}")
            df = None
        return df

    def save_data(self, base: str, df: pd.DataFrame):
        """ Saves complete data that is expected to be in 'df' and overwrites all previous content.
        """
        try:
            fname = self.fname(base)
            if isinstance(df, pd.Series):
                df = df.to_frame()
            df.to_hdf(self.fname(base), base + "_" + self.mnemonic(), mode="w")
            logger.info(f"saved {fname} {len(df)} samples from {df.index[0]} until {df.index[-1]}")
        except IOError:
            logger.error(f"cannot save {self.fname(base)}")

    def set_type_data(self, base: str, set_type: str):
        logger.debug(f"{type(self)} set type {set_type} of {base}")
        start_time = timeit.default_timer()
        if self.sets_split is None:
            try:
                self.sets_split = pd.read_csv(env.sets_split_fname(), skipinitialspace=True, sep="\t")
            except IOError:
                logger.error(f"pd.read_csv({env.sets_split_fname()}) IO error")
                return None
            # logger.debug(str(self.sets_split))

        sdf = self.sets_split.loc[self.sets_split.set_type == set_type]
        all = list()
        set_type_df = None
        for ix in sdf.index:
            last = pd.Timestamp(sdf.loc[ix, "end"])
            first = pd.Timestamp(sdf.loc[ix, "start"])
            df = self.get_data(base, first, last, use_cache=True)
            if (df is not None) and (not df.empty):
                all.append(df)
        if len(all) > 1:
            set_type_df = pd.concat(all, join="outer", axis=0)
        elif len(all) == 1:
            set_type_df = all[0]
        tdiff = (timeit.default_timer() - start_time) / 60
        logger.debug(f"{type(self)} set type {set_type} of {base} time: {tdiff:.0f} min")
        return set_type_df

    def check_report_obsolete(self, base):
        """ request data that crosses the boundry of to be downloaded and cached ohlcv data.
            Reports check results.
        """
        df = self.get_data(base, pd.Timestamp("2019-02-28 01:00:00+00:00"), 1000)
        logger.debug("{} {} history: {}, {} keys: {}, fname: {}".format(
            self.mnemonic(), base, self.history(), len(self.keys()), self.keys(), self.fname(base)))
        logger.debug(f"{self.mnemonic()} {base} got data: {len(df)} samples from {df.index[0]} until {df.index[-1]}")
        no_index_gaps(df)


class Ohlcv(CryptoData):
    """ Class for binance OHLCV samples handling
    """

    def history(self):
        "no history minutes are requires to download OHLCV data"
        return 0

    def keys(self):
        "returns the list of element keys"
        return Xch.data_keys

    def mnemonic(self):
        "returns a string that represent the PredictionData class as mnemonic, e.g. to use it in file names"
        return "OHLCV"

    def new_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        """ Predicts all samples and returns the result.
            set_type specific evaluations can be done using the saved prediction data.
            If use_cache == False then no saved data is used.
        """
        assert first is not None
        assert last is not None
        assert first <= last
        minutes = int((last - first) / pd.Timedelta(1, unit="T"))
        if minutes > 0:
            df = Xch.get_ohlcv(base, minutes, last)
            return df[Xch.data_keys]
        else:
            return None

    def get_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        """ Loads and downloads/calculates new data from 'first' sample up to and including 'last'.
            For Ohlcv enforce to use_cache.
        """
        return super().get_data(base, first, last, use_cache=True)


class Features(CryptoData):
    " Abstract superclass of all feature subclasses."

    def __init__(self, ohlcv: Ohlcv):
        self.ohlcv = ohlcv
        super().__init__()

    def get_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        """ Loads and downloads/calculates new data from 'first' sample up to and including 'last'.
            For Features enforce to use_cache.
        """
        return super().get_data(base, first, last, use_cache=True)


def check_df(df):
    # logger.debug(df.head())
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


def load_asset(bases):
    """ Loads the cached history data as well as live data from the xch
    """

    logger.debug(bases)  # Env.usage.bases)
    for base in bases:
        logger.debug(f"supplementing {base}")
        hdf = load_asset_dataframe(base, path=Env.data_path)
        # hdf.index.tz_localize(tz='UTC')

        last = (hdf.index[len(hdf)-1])
        # last = (hdf.index[len(hdf)-1]).tz_localize(tz='UTC')
        now = pd.Timestamp.utcnow()
        diffmin = int((now - last)/pd.Timedelta(1, unit='T'))
        ohlcv_df = Xch.get_ohlcv(base, diffmin, now)
        if ohlcv_df is None:
            logger.debug("skipping {}".format(base))
            continue
        tix = hdf.index[len(hdf)-1]
        if tix == ohlcv_df.index[0]:
            hdf = hdf.drop([tix])  # the last saved sample is incomple and needs to be updated
            # logger.debug("updated last sample of saved cache")
        hdf = pd.concat([hdf, ohlcv_df], sort=False)
        ok2save = check_df(hdf)
        if ok2save:
            save_asset_dataframe(hdf, base, path=Env.data_path)
        else:
            logger.debug(f"merged df checked: {ok2save} - dataframe not saved")


def create_testsets():
    ohlcv = Ohlcv()
    logger.debug(f"{ohlcv.path}")
    btc_ohlcv_df = ohlcv.get_data("btc", pd.Timestamp("2017-12-01 00:00:00+00:00"),
                                  pd.Timestamp("2018-01-31 23:59:00+00:00"))
    show_verbose(btc_ohlcv_df)
    xrp_ohlcv_df = ohlcv.get_data("xrp", pd.Timestamp("2017-12-01 00:00:00+00:00"),
                                  pd.Timestamp("2018-01-31 23:59:00+00:00"))
    show_verbose(xrp_ohlcv_df)
    env.test_mode()
    ohlcv2 = Ohlcv()
    logger.debug(f"{ohlcv2.path}")
    ohlcv2.save_data("btc", btc_ohlcv_df)
    ohlcv2.save_data("xrp", xrp_ohlcv_df)


def repair_ohlcv_from_old(bases):
    for base in bases:
        df = load_asset_dataframe(base, path=Env.data_path, limit=None)
        ohlcv = Ohlcv()
        ohlcv.save_data(base, df)


if __name__ == "__main__":
    # env.test_mode()
    # load_asset(Env.usage.bases)
    cd = CryptoData()
    create_testsets()
    # repair_ohlcv_from_old(["btc", "xrp"])
