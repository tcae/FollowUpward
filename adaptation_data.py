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
        return df_list

    @classmethod
    def set_type_data(self, bases: list, set_type: str, data_objs):
        if (data_objs is None) or (len(data_objs) == 0):
            logger.warning("missing data objects")
            return None
        if (bases is None) or (len(bases) == 0):
            logger.warning("missing bases")
            return None
        all_bases = list()
        for base in bases:
            sdf = SplitSets.set_type_datetime_ranges(set_type)
            start = sdf.start.min()
            end = sdf.end.max()
            all_types = list()
            for do in data_objs:
                df_list = SplitSets.split_sets(set_type, do.get_data(base, start, end))
                df = pd.concat(df_list, join="outer", axis=0)
                all_types.append(df)
            df = pd.concat(all_types, axis=1, join="inner")  # , keys=[do.mnemonic() for do in data_objs])
            all_bases.append(df)
        data_df = pd.concat(all_bases, join="outer", axis=0)  # , keys=bases)
        return data_df


class TrainingData:

    def __init__(self, features: ccd.Features, targets: ct.Targets):
        self.features = features
        self.targets = targets
        self.bgs = 10000  # batch group size
        self.tbgdf = None
        self.fbgdf = None
        self.bgdf_ix = 0

    def training_batch_size(self):
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

    def create_training_datasets(self):
        data_df = self.prepare_data()
        self.save_data(data_df)

    def prepare_data(self):
        """ Loads target and feature data per base and returns a data frame
            with all features and targets that is shuffled across bases
            as well as a corresponding counter data frame.
            Features and targets are distinguished via a multiindex level.
            Return value: tuple of (data df, counter df)
        """

        data_df = SplitSets.set_type_data(Env.bases, TRAIN, [self.features, self.targets])

        ixl = np.arange(len(data_df))
        np.random.shuffle(ixl)  # shuffle training samples across bases
        data_df = data_df.iloc[ixl]
        return data_df

    def fname(self):
        fname = self.targets.path + self.features.mnemonic() + "_" \
                + self.targets.mnemonic() + "_training.h5"
        return fname

    def load_index(self):
        """ Returns the complete shuffled index with the columns "base" and "timestamp"
        """
        fname = self.fname()
        with pd.HDFStore(fname, "r") as hdf:
            try:
                idf = hdf.get("/index")
                # idf = pd.read_hdf(fname, "/index")
            except IOError:
                logger.error(f"load data frame file ERROR: cannot load (base, timestamp) 'index' from {fname}")
                idf = None
        return idf

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

    def save_data(self, data_df):
        """ Saves a training data frame independent of base as well as the corresponding counters and metadata.
            This provides the means to shuffle across bases and load it in batches as needed.
        """
        fname = self.fname()

        with pd.HDFStore(fname, "w") as hdf:
            bs = self.training_batch_size()
            batches = int(math.ceil(len(data_df) / bs))
            meta_df = pd.Series(
                {"batch_size": bs, "batches": batches, "samples": len(data_df)})
            logger.debug("writing {} for {} samples as {} batches".format(
                fname, len(data_df), batches))
            hdf.put("/meta", meta_df, "fixed")
            # meta_df.to_hdf(fname, "/meta", mode="w")
            idf = data_df.index.to_frame(name=["base", "timestamp"], index=False)
            idf.index.set_names("idx", inplace=True)
            hdf.put("/index", idf, "fixed")
            # idf.to_hdf(fname, "/index", mode="a")
            data_df = data_df.reset_index(drop=True)
            data_df.index.set_names("idx", inplace=True)
            batch_groups = int(math.ceil(batches / self.bgs))
            bgs = bs * self.bgs
            check = 0
            logger.debug(f"bgroups: {batch_groups} * bgs = {batch_groups*self.bgs}")

            for bg in range(batch_groups - 1):
                sdf = data_df.iloc[bg * bgs: (bg + 1) * bgs]
                check += len(sdf)
                hdf.put(f"/features_{bg}", sdf["features"], "fixed")
                # sdf["features"].to_hdf(fname, f"/features_{bg}", mode="a")
                hdf.put(f"/targets_{bg}", sdf["targets"], "fixed")
                # sdf["targets"].to_hdf(fname, f"/targets_{bg}", mode="a")
            bg = batch_groups - 1
            sdf = data_df.iloc[bg * bgs:]
            check += len(sdf)
            hdf.put(f"/features_{bg}", sdf["features"], "fixed")
            # sdf["features"].to_hdf(fname, f"/features_{bg}", mode="a")
            hdf.put(f"/targets_{bg}", sdf["targets"], "fixed")
            # sdf["targets"].to_hdf(fname, f"/targets_{bg}", mode="a")
        assert check == len(data_df)


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
        self.scaler = scaler
        self.shuffle = shuffle
        self.training = TrainingData(features, targets)
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
        return prepare4keras(self.scaler, fdf, tdf, self.targets)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # logger.debug("TrainGenerator: on_epoch_end")
        if self.shuffle is True:
            # np.random.shuffle(self.indices)
            pass


class BaseGenerator(keras.utils.Sequence):
    'Generates validation data for Keras'
    def __init__(self, set_type, scaler, features: ccd.Features, targets: ct.Targets):
        # logger.debug("BaseGenerator: __init__")
        self.set_type = set_type
        self.scaler = scaler
        self.features = features
        self.targets = targets

    def __len__(self):
        'Denotes the number of batches per epoch'
        # logger.debug(f"BaseGenerator: __len__: {len(Env.bases)}")
        return len(Env.bases)

    def __getitem__(self, index):
        'Generate one batch of data'
        # logger.debug(f"BaseGenerator: __getitem__: {index}")
        if index >= len(Env.bases):
            logger.debug(f"BaseGenerator: index {index} > len {len(Env.bases)}")
            index %= len(Env.bases)
        base = Env.bases[index]
        fdf = SplitSets.set_type_data([base], self.set_type, [self.features])
        tdf = SplitSets.set_type_data([base], self.set_type, [self.targets])
        [fdf, tdf] = ccd.common_timerange([fdf, tdf])
        if (fdf is None) or fdf.empty:
            logger.warning("BaseGenerator: missing features")
        return prepare4keras(self.scaler, fdf, tdf, self.targets)


class AssessmentGenerator(keras.utils.Sequence):
    'Generates close, features, targets dataframes - use values attribute for Keras'
    def __init__(self, set_type, scaler, ohlcv: ccd.Ohlcv, features: ccd.Features, targets: ct.Targets):
        logger.debug("AssessmentGenerator: __init__")
        self.set_type = set_type
        self.scaler = scaler
        self.ohlcv = ohlcv
        self.features = features
        self.targets = targets

    def __len__(self):
        'Denotes the number of batches per epoch'
        logger.debug(f"AssessmentGenerator: __len__: {len(Env.bases)}")
        return len(Env.bases)

    def __getitem__(self, index):
        'Generate one batch of data'
        logger.debug(f"AssessmentGenerator: __getitem__: {index}")
        if index >= len(Env.bases):
            logger.debug(f"AssessmentGenerator: index {index} > len {len(Env.bases)}")
            index %= len(Env.bases)
        base = Env.bases[index]
        odf = SplitSets.set_type_data([base], self.set_type, [self.ohlcv])
        fdf = SplitSets.set_type_data([base], self.set_type, [self.features])
        tdf = SplitSets.set_type_data([base], self.set_type, [self.targets])
        [odf, fdf, tdf] = ccd.common_timerange([odf, fdf, tdf])
        # features, targets =  prepare4keras(self.scaler, fdf, tdf, self.tcls)
        return odf, fdf, tdf


if __name__ == "__main__":
    # tee = env.Tee()
    env.test_mode()
    ohlcv = ccd.Ohlcv()
    targets = ct.T10up5low30min(ohlcv)
    if True:
        features = cof.F2cond20(ohlcv)
    else:
        features = agf.F1agg110(ohlcv)
    td = TrainingData(features, targets)
    if False:  # create training sets
        start_time = timeit.default_timer()
        logger.debug("creating training batches")
        td.create_training_datasets()
        tdiff = (timeit.default_timer() - start_time)
        meta = td.load_meta()
        cnt = meta["samples"]
        logger.info(f"training set retrieval time: {(tdiff / 60):.0f} min = {(tdiff / cnt):.0f}s/sample")
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
    # tee.close()
    # SplitSets.set_type_datetime_ranges(TRAIN)
