import math
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


NA = "not assigned"
TRAIN = "training"
VAL = "validation"
TEST = "test"


class TrainingData:

    def __init__(self, features: ccd.Features, targets: ct.Targets):
        self.features = features
        self.targets = targets
        self.bgs = 100  # batch group size
        self.tbgdf = None
        self.fbgdf = None
        self.bgdf_ix = 0
        self.hdf = None

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
        all = list()
        for base in Env.bases:
            fbdf = self.features.set_type_data(base, "training")
            tbdf = self.targets.set_type_data(base, "training")
            bdf = pd.concat([fbdf, tbdf], axis=1, join="inner", keys=["features", "targets"])
            all.append(bdf)
        data_df = pd.concat(all, join="outer", axis=0, keys=Env.bases)

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
        if self.hdf is None:
            self.hdf = pd.HDFStore(fname, "r")
        try:
            idf = self.hdf.get("/index")
            # idf = pd.read_hdf(fname, "/index")
            return idf
        except IOError:
            print(f"{env.timestr()} load data frame file ERROR: cannot load (base, timestamp) 'index' from {fname}")
            return None

    def load_meta(self):
        "Returns the metadata batch_size, batches, samples as a dictionary"
        fname = self.fname()
        if self.hdf is None:
            self.hdf = pd.HDFStore(fname, "r")
        try:
            mdf = self.hdf.get("/meta")
            # mdf = pd.read_hdf(fname, "/meta")
            meta = {ix: mdf[ix] for ix in mdf.index}
            return meta
        except IOError:
            print(f"{env.timestr()} load data frame file ERROR: cannot load 'meta' from {fname}")
            return None

    def load_data(self, batch_ix):
        """ Returns a data frame for all saved training data independent of base.
            This provides the means to shuffle across bases and load it in batches as needed.
        """
        if self.hdf is None:
            self.hdf = pd.HDFStore(self.fname(), "r")
        try:
            fname = self.fname()
            bg_ix = int(batch_ix / self.bgs)
            if (self.tbgdf is None) or (self.bgdf_ix != bg_ix):
                self.bgdf_ix = bg_ix
                self.fbgdf = self.hdf.get(f"/features_{bg_ix}")
                self.tbgdf = self.hdf.get(f"/targets_{bg_ix}")
                # self.fbgdf = fdf = pd.read_hdf(fname, f"/features_{bg_ix}")
                # self.tbgdf = tdf = pd.read_hdf(fname, f"/targets_{bg_ix}")
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
        except IOError:
            print(f"{env.timestr()} load data frame file ERROR: cannot load batch_ix {batch_ix} from {fname}")
            return None, None

    def save_data(self, data_df):
        """ Saves a training data frame independent of base as well as the corresponding counters and metadata.
            This provides the means to shuffle across bases and load it in batches as needed.
        """
        fname = self.fname()
        assert self.hdf is None

        hdf = pd.HDFStore(fname, "w")
        bs = self.training_batch_size()
        batches = int(math.ceil(len(data_df) / bs))
        meta_df = pd.Series(
            {"batch_size": bs, "batches": batches, "samples": len(data_df)})
        print("{}: writing {} for {} samples as {} batches".format(
            env.timestr(), fname, len(data_df), batches))
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
        print(f"bgroups: {batch_groups} * bgs = {batch_groups*self.bgs}")

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
        hdf.close()
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
        return self.batches

    def __getitem__(self, index):
        "Generate one batch of data"
        if index >= self.batches:
            print(f"TrainGenerator: index {index} > len {self.batches}")
            index %= self.batches
        fdf, tdf = self.training.load_data(index)
        return prepare4keras(self.scaler, fdf, tdf, self.targets)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        print("TrainGenerator: on_epoch_end")
        if self.shuffle is True:
            # np.random.shuffle(self.indices)
            pass


class ValidationGenerator(keras.utils.Sequence):
    'Generates validation data for Keras'
    def __init__(self, set_type, scaler, features: ccd.Features, targets: ct.Targets):
        self.set_type = set_type
        self.scaler = scaler
        self.features = features
        self.targets = targets

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(Env.bases)

    def __getitem__(self, index):
        'Generate one batch of data'
        if index >= len(Env.bases):
            print(f"BaseGenerator: index {index} > len {len(Env.bases)}")
            index %= len(Env.bases)
        base = Env.bases[index]
        fdf = self.features.set_type_data(base, self.set_type)
        tdf = self.targets.set_type_data(base, self.set_type)
        [fdf, tdf] = ccd.common_timerange([fdf, tdf])
        return prepare4keras(self.scaler, fdf, tdf, self.targets)


class AssessmentGenerator(keras.utils.Sequence):
    'Generates close, features, targets dataframes - use values attribute for Keras'
    def __init__(self, set_type, scaler, ohlcv: ccd.Ohlcv, features: ccd.Features, targets: ct.Targets):
        self.set_type = set_type
        self.scaler = scaler
        self.ohlcv = ohlcv
        self.features = features
        self.targets = targets

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(Env.bases)

    def __getitem__(self, index):
        'Generate one batch of data'
        if index >= len(Env.bases):
            print(f"BaseGenerator: index {index} > len {len(Env.bases)}")
            index %= len(Env.bases)
        base = Env.bases[index]
        odf = self.ohlcv.set_type_data(base, self.set_type)
        fdf = self.features.set_type_data(base, self.set_type)
        tdf = self.targets.set_type_data(base, self.set_type)
        [odf, fdf, tdf] = ccd.common_timerange([odf, fdf, tdf])
        # features, targets =  prepare4keras(self.scaler, fdf, tdf, self.tcls)
        return odf, fdf, tdf


if __name__ == "__main__":
    tee = env.Tee()
    ohlcv = ccd.Ohlcv()
    targets = ct.T10up5low30min(ohlcv)
    if True:
        features = cof.F2cond20(ohlcv)
    else:
        features = agf.F1agg110(ohlcv)
    td = TrainingData(features, targets)
    if False:  # create training sets
        start_time = timeit.default_timer()
        print(f"{env.timestr()} creating training batches")
        td.create_training_datasets()
        tdiff = (timeit.default_timer() - start_time)
        meta = td.load_meta()
        cnt = meta["samples"]
        print(f"{env.timestr()} training set retrieval time: {(tdiff / 60):.0f} min = {(tdiff / cnt):.0f}s/sample")
    if False:  # retrieve training sets for test purposes
        meta = td.load_meta()
        print(meta)
        start_time = timeit.default_timer()
        print(f"{env.timestr()} loading training batches")
        # bix = 137339
        # fdf, tdf = td.load_data(bix)
        # print(f"ix: {bix}  len: {len(fdf)} {fdf}")
        check = 0
        for bix in range(meta["batches"]):
            fdf, tdf = td.load_data(bix)
            check += len(fdf)
        print(f"sample sum loaded: {check}")
        tdiff = (timeit.default_timer() - start_time)
        print(f"{env.timestr()} training set retrieval time: {(tdiff):.0f} s = {tdiff/check:.0f}s/sample")
        # idf = td.load_index()
        # print(idf.describe(percentiles=[], include='all'))
    tee.close()
