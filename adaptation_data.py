import math
import logging
# import datetime
import timeit
import pandas as pd
import numpy as np
# import tables as pt
from sklearn.utils import Bunch
import tensorflow.keras as keras
import env_config as env
from env_config import Env
import cached_crypto_data as ccd
import crypto_targets as ct
# import crypto_features as cf
import condensed_features as cof
import aggregated_features as agf

logger = logging.getLogger(__name__)

NA = "not assigned"
TRAIN = "training"
VAL = "validation"
TEST = "test"


class SplitSets:
    sets_split = None

    @classmethod
    def overall_timerange(cls):
        """ Returns a data frame with 'start' and 'end' column of given set_type.
        """
        if cls.sets_split is None:
            try:
                cls.sets_split = pd.read_csv(env.sets_split_fname(), skipinitialspace=True, sep="\t",
                                             parse_dates=["start", "end"])
            except IOError:
                logger.error(f"pd.read_csv({env.sets_split_fname()}) IO error")
                return None
            # logger.debug(str(cls.sets_split))

        first = cls.sets_split.start.min()
        last = cls.sets_split.end.max()
        logger.debug(f"split set time overall start {first} - end {last}")
        return (first, last)

    @classmethod
    def set_type_datetime_ranges(cls, set_type: str):
        """ Returns a data frame with 'start' and 'end' column of given set_type.
        """
        if cls.sets_split is None:
            try:
                cls.sets_split = pd.read_csv(env.sets_split_fname(), skipinitialspace=True, sep="\t",
                                             parse_dates=["start", "end"])
            except IOError:
                logger.error(f"pd.read_csv({env.sets_split_fname()}) IO error")
                return None
            # logger.debug(str(cls.sets_split))

        sdf = cls.sets_split.loc[cls.sets_split.set_type == set_type]
        # logger.debug(f"split set time ranges \n{sdf}")
        return sdf

    @classmethod
    def split_sets(cls, set_type: str, df: pd.DataFrame):
        sdf = cls.set_type_datetime_ranges(set_type)
        df_list = [df.loc[(df.index >= pd.Timestamp(sdf.loc[ix, "start"])) &
                          (df.index <= pd.Timestamp(sdf.loc[ix, "end"]))] for ix in sdf.index]
        df_list = [df for df in df_list if (df is not None) and (not df.empty)]  # remove None and empty df
        return df_list

    @classmethod
    def set_type_data(cls, base: str, set_type: str, ohlcv: ccd.Ohlcv, features: ccd.Features, targets: ct.Targets):
        sdf = SplitSets.set_type_datetime_ranges(set_type)
        start = sdf.start.min()
        end = sdf.end.max()
        ohlcv_df = features_df = targets_df = None
        if ohlcv is not None:
            df_list = SplitSets.split_sets(set_type, ohlcv.get_data(base, start, end))
            ohlcv_df = pd.concat(df_list, join="outer", axis=0)
            if (ohlcv_df is None) or ohlcv_df.empty:
                logger.warning(f"unexpected missing {base} ohlcv {set_type} data")
        if features is not None:
            df_list = SplitSets.split_sets(set_type, features.get_data(base, start, end))
            features_df = pd.concat(df_list, join="outer", axis=0)
            if (features_df is None) or features_df.empty:
                logger.warning(f"unexpected missing {base} features {set_type} data")
        if targets is not None:
            df_list = SplitSets.split_sets(set_type, targets.get_data(base, start, end))
            targets_df = pd.concat(df_list, join="outer", axis=0)
            if (targets_df is None) or targets_df.empty:
                logger.warning(f"unexpected missing {base} targets {set_type} data")
        return ohlcv_df, features_df, targets_df


class TrainingData:

    def __init__(self, bases: list, scaler: dict, features: ccd.Features, targets: ct.Targets):
        self.features = features
        self.targets = targets
        self.scaler = scaler  # dict: key == base str, value == scaler object
        self.bgs = 10000  # batch group size
        self.tbgdf = None  # target batch group data frame
        self.fbgdf = None  # feature batch group data frame
        self.bgdf_ix = 0
        self.bases = bases

    @classmethod
    def training_batch_size(cls):
        """ Return the number of training samples for one batch.
        """
        return 32

    # def _collect_counter(self, base, df):
    #     td = self.targets.target_dict()
    #     counter = {td[key]: len(df.loc[df[("targets", "target")] == td[key]]) for key in td}
    #     counter["unknown"] = len(df) - sum(counter.values())
    #     counter["total"] = len(df)
    #     counter["base"] = base
    #     return counter

    # def counter_df(self):
    #     ckeys = [key for key in td]
    #     ckeys += ["unknown", "total", "base"]
    #     cdf = pd.DataFrame(columns=ckeys)
    #     return cdf

    def create_type_datasets(self, set_type: str, label: str = None):
        """ Loads target and feature data per base and stores data frames
            with all features and targets that is scaled across bases.
        """
        fdfl = list()
        tdfl = list()
        lfdf = 0
        ltdf = 0
        for base in self.bases:
            _, fdf, tdf = SplitSets.set_type_data(base, set_type, None, self.features, self.targets)
            if label is not None:
                [fdf, tdf] = ccd.common_timerange([fdf, tdf])
                fdf = fdf.loc[tdf.target == ct.TARGETS[label]]
                tdf = tdf.loc[tdf.target == ct.TARGETS[label]]
                assert len(tdf) == len(fdf)
            # logger.info(
            #     "before scaling {} {} first {} last {}\n{}".format(
            #         base, self.features.mnemonic(), fdf.index[0], fdf.index[-1],
            #         fdf.describe(percentiles=[], include='all')))
            if base in self.scaler:
                fdf = pd.DataFrame(index=fdf.index, columns=fdf.columns, data=self.scaler[base].transform(fdf.values))
                # logger.info(
                #     "after scaling {} {} first {} last {}\n{}".format(
                #         base, self.features.mnemonic(), fdf.index[0], fdf.index[-1],
                #         fdf.describe(percentiles=[], include='all')))
            else:
                logger.error(f"missing scaler for {base}")
            fdfl.append(fdf)
            lfdf += len(fdf)
            tdfl.append(tdf)
            ltdf += len(tdf)
        fdf = pd.concat(fdfl, join="outer", axis=0, ignore_index=True)
        if len(fdf) == lfdf:
            logger.info(f"features concatenated {len(fdf)}")
        else:
            logger.error(f"features concatenated {len(fdf)} != sum of part {lfdf}")

        tdf = pd.concat(tdfl, join="outer", axis=0, ignore_index=True)
        if len(tdf) == ltdf:
            logger.info(f"targets concatenated {len(tdf)}, diff targets-features = {len(tdf) - len(fdf)}")
        else:
            logger.error(f"targets concatenated {len(tdf)} != sum of part {ltdf}")

        [fdf, tdf] = ccd.common_timerange([fdf, tdf])
        assert len(fdf) == len(tdf)

        fdf.index.set_names("idx", inplace=True)
        tdf.index.set_names("idx", inplace=True)
        return (fdf, tdf)

    def create_training_datasets(self, label: str = None):
        """ Loads target and feature data per base and stores data frames
            with all features and targets that is scaled and shuffled across bases.
        """
        (fdf, tdf) = self.create_type_datasets(TRAIN, label)

        ixl = np.arange(len(fdf))
        np.random.shuffle(ixl)  # shuffle training samples across bases
        fdf = fdf.iloc[ixl]
        tdf = tdf.iloc[ixl]
        return (fdf, tdf)

    def fname(self):
        fname = self.targets.path + self.features.mnemonic() + "_" \
                + self.targets.mnemonic() + "_training.h5"
        return fname

    def load_meta(self):
        "Returns the metadata batch_size, batches, samples as a dictionary"
        fname = self.fname()
        with pd.HDFStore(fname, "r") as hdf:
            try:
                mdf = hdf.get("/meta")
                # mdf = pd.read_hdf(fname, "/meta")
                meta = {ix: mdf[ix] for ix in mdf.index}
            except IOError:
                logger.error(f"load data frame file ERROR: cannot load 'meta' from {fname}")
                meta = None
        return meta

    def load_data(self, batch_ix):
        """ Returns a data frame for all saved training data independent of base.
            This provides the means to shuffle across bases and load it in batches as needed.
        """
        bg_ix = int(batch_ix / self.bgs)
        if (self.fbgdf is None) or (self.tbgdf is None) or (self.bgdf_ix != bg_ix):
            # expensive file access is only required on batch group change;
            # batches within the same group are already loaded in fbgdf/tbgdf
            with pd.HDFStore(self.fname(), "r") as hdf:
                try:
                    self.bgdf_ix = bg_ix
                    self.fbgdf = hdf.get(f"/features_{bg_ix}")
                    self.tbgdf = hdf.get(f"/targets_{bg_ix}")
                except IOError:
                    logger.error("load data frame file ERROR: cannot load batch_ix {} from {}".format(
                        batch_ix, self.fname()))
                    return None, None
        tdf = self.tbgdf
        fdf = self.fbgdf
        bs = self.training_batch_size()
        ix = (batch_ix % self.bgs) * bs
        if ix + bs > len(fdf):
            fdf = fdf.iloc[ix:]
            tdf = tdf.iloc[ix:]
        else:
            fdf = fdf.iloc[ix:ix+bs]
            tdf = tdf.iloc[ix:ix+bs]
        return fdf, tdf

    def save_data(self, features_df, targets_df):
        """ Saves a training data frame independent of base as well as the corresponding counters and metadata.
            This provides the means to shuffle across bases and load it in batches as needed.

            groups of shuffled are used as batch to trian one weight adjustment step
            batch groups are stored together to balance expensive file access operations against required RAM memory
        """
        fname = self.fname()
        assert len(features_df) == len(targets_df)
        with pd.HDFStore(fname, "w") as hdf:
            bs = self.training_batch_size()
            batches = int(math.ceil(len(features_df) / bs))
            meta_df = pd.Series(
                {"batch_size": bs, "batches": batches, "samples": len(features_df)})
            logger.debug("writing {} for {} samples as {} batches".format(
                fname, len(features_df), batches))
            hdf.put("/meta", meta_df, "fixed")
            # meta_df.to_hdf(fname, "/meta", mode="w")
            batch_groups = int(math.ceil(batches / self.bgs))
            bgs = bs * self.bgs
            fcheck = 0
            tcheck = 0
            logger.debug(f"bgroups: {batch_groups} * bgs = {batch_groups*self.bgs}")

            for bg in range(batch_groups - 1):
                sfdf = features_df.iloc[bg * bgs: (bg + 1) * bgs]
                stdf = targets_df.iloc[bg * bgs: (bg + 1) * bgs]
                fcheck += len(sfdf)
                tcheck += len(stdf)
                hdf.put(f"/features_{bg}", sfdf, "fixed")
                # sfdf.to_hdf(fname, f"/features_{bg}", mode="a")
                hdf.put(f"/targets_{bg}", stdf, "fixed")
                # stdf.to_hdf(fname, f"/targets_{bg}", mode="a")
            bg = batch_groups - 1
            sfdf = features_df.iloc[bg * bgs:]
            stdf = targets_df.iloc[bg * bgs:]
            fcheck += len(sfdf)
            tcheck += len(stdf)
            hdf.put(f"/features_{bg}", sfdf, "fixed")
            # sfdf.to_hdf(fname, f"/features_{bg}", mode="a")
            hdf.put(f"/targets_{bg}", stdf, "fixed")
            # stdf.to_hdf(fname, f"/targets_{bg}", mode="a")
        assert fcheck == len(features_df)
        assert tcheck == len(targets_df)


def subset_check(base: str, ix: int, acd: ccd.CryptoData, acdf: pd.DataFrame, bcd: ccd.CryptoData, bcdf: pd.DataFrame):
    if (acdf is None) or (len(acdf) == 0):
        logger.warning(f"no {acd.mnemonic()} data for subset {ix} of {base}")
        return
    if (bcdf is None) or (len(bcdf) == 0):
        logger.warning(f"no {bcd.mnemonic()} data for subset {ix} of {base}")
        return
    if acdf.index[-1] != bcdf.index[-1]:
        logger.warning("unexpected {}/{} last timestamp difference {} {} vs. {} {}".format(
                        base, ix, acd.mnemonic(), acdf.index[-1], bcd.mnemonic(), bcdf.index[-1]))
    if (acdf.index[0] != bcdf.index[0]) and \
       (acdf.index[0] != (bcdf.index[0] - pd.Timedelta(bcd.history(), unit="T"))):
        logger.warning(
            "unexpected {}/{} first timestamp difference {} {} vs. {} {} - history {}".format(
                base, ix, acd.mnemonic(), acdf.index[0], bcd.mnemonic(), bcdf.index[0],
                (bcdf.index[0] - pd.Timedelta(bcd.history(), unit="T"))))
    if (len(acdf) != len(bcdf)) and (len(acdf) != (len(bcdf) + bcd.history())):
        logger.warning(
            "unexpected {}/{} length {} {} vs. {}  {} + history {}".format(
                base, ix, acd.mnemonic(), len(acdf), bcd.mnemonic(), len(bcdf), len(bcdf) + bcd.history()))


def set_type_check(ohlcv: ccd.Ohlcv, features: ccd.Features, targets: ct.Targets, bases: list, set_type: str):
    logger.info(f"split set timeranges \n{SplitSets.set_type_datetime_ranges(set_type).reset_index()}")
    for base in bases:
        odf = ohlcv.load_data(base)
        fdf = features.load_data(base)
        tdf = targets.load_data(base)
        logger.info(
            "{} {} first {}, {} first {}, {} first {}, last {}".format(
                base, ohlcv.mnemonic(), odf.index[0],
                features.mnemonic(), fdf.index[0], targets.mnemonic(), tdf.index[0], odf.index[-1]))
        if ((odf.index[-1]-odf.index[0])/pd.Timedelta(1, unit="T") + 1 - len(odf)) != 0:
            logger.warning(
                "{} {} gap detected: min diff = {} vs len = {}".format(
                    base, ohlcv.mnemonic(), (odf.index[-1]-odf.index[0])/pd.Timedelta(1, unit="T")+1, len(odf)))
        if ((fdf.index[-1]-fdf.index[0])/pd.Timedelta(1, unit="T") + 1 - len(fdf)) != 0:
            logger.warning(
                "{} {} gap detected: min diff = {} vs len = {}".format(
                    base, features.mnemonic(), (fdf.index[-1]-fdf.index[0])/pd.Timedelta(1, unit="T")+1, len(fdf)))
        if ((tdf.index[-1]-tdf.index[0])/pd.Timedelta(1, unit="T") + 1 - len(tdf)) != 0:
            logger.warning(
                "{} {} gap detected: min diff = {} vs len = {}".format(
                    base, targets.mnemonic(), (tdf.index[-1]-tdf.index[0])/pd.Timedelta(1, unit="T")+1, len(tdf)))
        # tdiff = (timeit.default_timer() - start_time2) / 60
        # logger.debug(f"prediction data {base} time: {tdiff:.1f} min")

        # start_time2 = timeit.default_timer()
        odfl = SplitSets.split_sets(set_type, odf)
        fdfl = SplitSets.split_sets(set_type, fdf)
        tdfl = SplitSets.split_sets(set_type, tdf)
        if len(odfl) != len(fdfl):
            logger.warning("unexpected {} subset differnce ohlcv {} vs. features {}".format(
                base, len(odfl), len(fdfl)))
        if len(odfl) != len(tdfl):
            logger.warning("unexpected {} subset differnce ohlcv {} vs. targets {}".format(
                base, len(odfl), len(tdfl)))
        logger.info(f"checking {len(odfl)} {set_type} subsets of {base}")
        for ix in range(len(odfl)):
            [odf, fdf, tdf] = [odfl[ix], fdfl[ix], tdfl[ix]]
            if (odf is None) or (len(odf) == 0):
                logger.info(f"no ohlcv data for subset {ix} of {base}")
                continue
            subset_check(base, ix, ohlcv, odf, features, fdf)
            subset_check(base, ix, ohlcv, odf, targets, tdf)


def set_check(ohlcv: ccd.Ohlcv, features: ccd.Features, targets: ct.Targets, bases: list):
    for base in bases:
        odf = ohlcv.load_data(base)
        ccd.no_index_gaps(odf)
        logger.info(
            "{} {} first {} last {}\n{}".format(
                base, ohlcv.mnemonic(), odf.index[0], odf.index[-1], odf.describe(percentiles=[], include='all')))
        if ((odf.index[-1]-odf.index[0])/pd.Timedelta(1, unit="T") + 1 - len(odf)) != 0:
            logger.warning(
                "{} {} gap detected: min diff = {} vs len = {}".format(
                    base, ohlcv.mnemonic(), (odf.index[-1]-odf.index[0])/pd.Timedelta(1, unit="T")+1, len(odf)))
        fdf = features.load_data(base)
        logger.info(
            "{} {} first {} last {}\n{}".format(
                base, features.mnemonic(), fdf.index[0], fdf.index[-1], fdf.describe(percentiles=[], include='all')))
        if ((fdf.index[-1]-fdf.index[0])/pd.Timedelta(1, unit="T") + 1 - len(fdf)) != 0:
            logger.warning(
                "{} {} gap detected: min diff = {} vs len = {}".format(
                    base, features.mnemonic(), (fdf.index[-1]-fdf.index[0])/pd.Timedelta(1, unit="T")+1, len(fdf)))
        if ((odf.index[0]-fdf.index[0])/pd.Timedelta(1, unit="T") != 0) and \
           ((odf.index[0]-fdf.index[0])/pd.Timedelta(1, unit="T") != features.history()):
            logger.warning(
                "{} {}/{} start diff {} != history {}".format(
                    base, ohlcv.mnemonic(), features.mnemonic(), (odf.index[-1]-fdf.index[0])/pd.Timedelta(1, unit="T"),
                    features.history()))
        tdf = targets.load_data(base)
        logger.info(
            "{} {} first {} last {}\n{}".format(
                base, targets.mnemonic(), tdf.index[0], tdf.index[-1], tdf.describe(percentiles=[], include='all')))
        ct.report_setsize(base, tdf)
        if ((tdf.index[-1]-tdf.index[0])/pd.Timedelta(1, unit="T") + 1 - len(tdf)) != 0:
            logger.warning(
                "{} {} gap detected: min diff = {} vs len = {}".format(
                    base, targets.mnemonic(), (tdf.index[-1]-tdf.index[0])/pd.Timedelta(1, unit="T")+1, len(tdf)))
        if ((odf.index[0]-tdf.index[0])/pd.Timedelta(1, unit="T") != 0) and \
           ((odf.index[0]-tdf.index[0])/pd.Timedelta(1, unit="T") != targets.history()):
            logger.warning(
                "{} {}/{} start diff {} != history {}".format(
                    base, ohlcv.mnemonic(), targets.mnemonic(), (odf.index[-1]-tdf.index[0])/pd.Timedelta(1, unit="T"),
                    targets.history()))


def prepare4keras(scaler, feature_df, target_df, targets):
    samples = Bunch(
        data=feature_df.values,
        target=target_df.values,
        target_names=np.array(targets.target_dict().keys()),
        feature_names=np.array(feature_df.keys()))
    if scaler is not None:
        samples.data = scaler.transform(samples.data)
    samples.target = keras.utils.to_categorical(
        samples.target, num_classes=len(targets.target_dict().keys()))
    return samples.data, samples.target


class TrainingGenerator(keras.utils.Sequence):
    "Generates training data for Keras"
    def __init__(self, scaler, features: ccd.Features, targets: ct.Targets, shuffle=False):
        # logger.debug("TrainGenerator: __init__")
        self.shuffle = shuffle
        self.training = TrainingData(scaler, features, targets)
        meta = self.training.load_meta()
        self.features = features
        self.targets = targets
        self.batches = meta["batches"]
        self.batch_size = meta["batch_size"]
        self.samples = meta["samples"]

    def __len__(self):
        "Denotes the number of batches per epoch"
        # logger.debug(f"TrainGenerator: __len__ :{self.batches}")
        return self.batches

    def __getitem__(self, index):
        "Generate one batch of data"
        # logger.debug(f"TrainGenerator: __getitem__ ix:{index}")
        if index >= self.batches:
            logger.debug(f"TrainGenerator: index {index} > len {self.batches}")
            index %= self.batches
        fdf, tdf = self.training.load_data(index)
        if (fdf is None) or fdf.empty:
            logger.warning("TrainGenerator: missing features")
        return prepare4keras(None, fdf, tdf, self.targets)  # no scaler as shuffled features are already scaled

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # logger.debug("TrainGenerator: on_epoch_end")
        if self.shuffle is True:
            # np.random.shuffle(self.indices)
            pass


class BaseGenerator(keras.utils.Sequence):
    'Generates validation data for Keras'
    def __init__(self, set_type: str, bases: list, scaler, features: ccd.Features, targets: ct.Targets):
        # logger.debug(f"BaseGenerator({set_type}): __init__")
        self.set_type = set_type
        self.scaler = scaler  # dict: key == base str, value == scaler object
        self.features = features
        self.targets = targets
        self.ixl = list()  # list of indices refering to a (base_ix, subset_ix) tuple
        self.bases = bases

    def __len__(self):
        'Denotes the number of batches per epoch'
        self.ixl = \
            [(bix, six) for bix, base in enumerate(self.bases) for six, subset_df in
             enumerate(SplitSets.split_sets(self.set_type, self.features.load_data(base)))]
        # features require the longest history -> use features as limiting data
        # logger.debug(f"BaseGenerator {self.set_type}: len = {len(self.ixl)}")
        return len(self.ixl)

    def __getitem__(self, index):
        'Generate one batch of data'
        if index >= len(self.ixl):
            logger.debug(f"BaseGenerator {self.set_type}: index {index} >= len {len(self.ixl)}")
            index %= len(self.bases)
        (bix, six) = self.ixl[index]
        base = self.bases[bix]
        # logger.debug(
        #     f"BaseGenerator {self.set_type}: __getitem__: index = {index} base = {base} subset index = {six}")
        if self.features is not None:
            df_list = SplitSets.split_sets(self.set_type, self.features.load_data(base))
            fdf = df_list[six]
            assert fdf is not None
        if self.targets is not None:
            df_list = SplitSets.split_sets(self.set_type, self.targets.load_data(base))
            tdf = df_list[six]
            assert tdf is not None
        fdf, tdf = ccd.common_timerange([fdf, tdf])

        if base in self.scaler:
            return prepare4keras(self.scaler[base], fdf, tdf, self.targets)
        else:
            logger.error(f"BaseGenerator {self.set_type}: no scaler for {base}")
            return prepare4keras(None, fdf, tdf, self.targets)


class AssessmentGenerator(keras.utils.Sequence):
    'Generates close, features, targets dataframes - use values attribute for Keras'
    def __init__(self, set_type: str, bases: list, ohlcv: ccd.Ohlcv, features: ccd.Features, targets: ct.Targets):
        # logger.debug(f"AssessmentGenerator {set_type}: __init__")
        self.set_type = set_type
        self.ohlcv = ohlcv
        self.features = features
        self.targets = targets
        self.bases = bases

    def __len__(self):
        'Denotes the number of batches per epoch'
        self.ixl = \
            [(bix, six) for bix, base in enumerate(self.bases) for six, subset_df in
             enumerate(SplitSets.split_sets(self.set_type, self.features.load_data(base)))]
        # features require the longest history -> use features as limiting data
        # logger.debug(f"AssessmentGenerator {self.set_type}: len = {len(self.ixl)}")
        return len(self.ixl)

    def __getitem__(self, index):
        'Generate one batch of data'
        if index >= len(self.ixl):
            logger.debug(f"AssessmentGenerator {self.set_type}: index {index} >= len {len(self.ixl)}")
            index %= len(self.bases)
        (bix, six) = self.ixl[index]
        base = self.bases[bix]
        # logger.debug(
        #     f"AssessmentGenerator {self.set_type}: __getitem__: index = {index} base = {base} subset index = {six}")
        if self.ohlcv is not None:
            df_list = SplitSets.split_sets(self.set_type, self.ohlcv.load_data(base))
            odf = df_list[six]
            assert odf is not None
        if self.features is not None:
            df_list = SplitSets.split_sets(self.set_type, self.features.load_data(base))
            fdf = df_list[six]
            assert fdf is not None
        if self.targets is not None:
            df_list = SplitSets.split_sets(self.set_type, self.targets.load_data(base))
            tdf = df_list[six]
            assert tdf is not None
        odf, fdf, tdf = ccd.common_timerange([odf, fdf, tdf])
        return base, odf, fdf, tdf


if __name__ == "__main__":
    # tee = env.Tee()
    # env.test_mode()
    ohlcv = ccd.Ohlcv()
    targets = ct.Targets(ohlcv)
    # targets = ct.T10up5low30min(ohlcv)
    if True:
        features = cof.F3cond14(ohlcv)
    else:
        features = agf.F1agg110(ohlcv)
    td = TrainingData(Env.bases, dict(), features, targets)
    if False:  # create training sets
        # ! don't use this part anymore because training data creation requires trained scaler
        start_time = timeit.default_timer()
        logger.debug("creating training batches")
        td.create_training_datasets()
        tdiff = (timeit.default_timer() - start_time)
        meta = td.load_meta()
        cnt = meta["samples"]
        logger.info(f"training set creation time: {(tdiff / 60):.0f} min = {(tdiff / cnt):.0f}s/sample")
    if False:  # retrieve training sets for test purposes
        meta = td.load_meta()
        logger.debug(str(meta))
        start_time = timeit.default_timer()
        logger.info("loading training batches")
        # bix = 137339
        # fdf, tdf = td.load_data(bix)
        # logger.debug(f"ix: {bix}  len: {len(fdf)} {fdf}")
        check = 0
        for bix in range(meta["batches"]):
            fdf, tdf = td.load_data(bix)
            check += len(fdf)
        logger.debug(f"sample sum loaded: {check}")
        tdiff = (timeit.default_timer() - start_time)
        logger.info(f"training set retrieval time: {(tdiff):.0f} s = {tdiff/check:.0f}s/sample")
        # idf = td.load_index()
        # logger.info(str(idf.describe(percentiles=[], include='all')))
    if True:
        set_check(ohlcv, features, targets, Env.bases)
        set_type_check(ohlcv, features, targets, Env.bases, TRAIN)
        set_type_check(ohlcv, features, targets, Env.bases, VAL)
        set_type_check(ohlcv, features, targets, Env.bases, TEST)
    # tee.close()
    # SplitSets.set_type_datetime_ranges(TRAIN)
