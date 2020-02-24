import math
import datetime
import pandas as pd
import numpy as np
from sklearn.utils import Bunch
import tensorflow.keras as keras
import env_config as env
from env_config import Env
import cached_crypto_data as ccd
import crypto_targets as ct
import crypto_features as cf
import condensed_features as cof
import aggregated_features as agf


class TrainingData:

    def __init__(self, target_class: ct.Targets, feature_class: cf.Features):
        self.tcls = target_class
        self.fcls = feature_class

    def training_batch_size(self):
        """ Return the number of training samples for one batch.
        """
        return 32

    def _collect_counter(self, base, df):
        td = self.tcls.target_dict()
        counter = {td[key]: len(df.loc[df[("targets", "target")] == td[key]]) for key in td}
        counter["unknown"] = len(df) - sum(counter.values())
        counter["total"] = len(df)
        counter["base"] = base
        return counter

    def create_training_datasets(self):
        (data_df, counter_df) = self.prepare_data()
        self.save_data(data_df, counter_df)

    def prepare_data(self):
        """ Loads target and feature data per base and returns a data frame
            with all features and targets that is shuffled across bases
            as well as a corresponding counter data frame.
            Features and targets are distinguished via a multiindex level.
            Return value: tuple of (data df, counter df)
        """
        all = list()
        counter_list = list()
        for base in Env.bases:
            fbdf = self.fcls.training_data(base)
            tbdf = self.tcls.training_data(base)
            bdf = pd.concat([fbdf, tbdf], axis=1, join="inner", keys=["features", "targets"])
            counter_list.append(self._collect_counter(base, bdf))  # collect list of dicts
            all.append(bdf)
        data_df = pd.concat(all, join="outer", axis=0, keys=Env.bases)

        all_counter = {basekey: [] for basekey in counter_list[0]}
        for basedict in counter_list:
            for basekey in all_counter:
                all_counter[basekey].append(basedict[basekey])
        # for basekey in all_counter:  # add a list entry with totals
        #     if basekey == "base":
        #         all_counter[basekey].append("total")
        #     else:
        #         all_counter[basekey].append(sum(all_counter[basekey]))
        counter_df = pd.DataFrame(all_counter)

        ixl = np.arange(len(data_df))
        np.random.shuffle(ixl)  # shuffle training samples across bases
        data_df = data_df.iloc[ixl]
        return (data_df, counter_df)

    def fname(self):
        fname = ccd.CryptoData.path + self.tcls.data_mnemonic() + "_" + self.fcls.data_mnemonic() + "_training.h5"
        return fname

    def load_data(self, batch_ix):
        """ Returns a data frame for all saved training data independent of base.
            This provides the means to shuffle across bases and load it in batches as needed.
        """
        try:
            fname = self.training_fname()
            df = pd.read_hdf(fname, batch_ix)
            return df["features"], df["targets"]
        except IOError:
            print(f"{env.timestr()} load data frame file ERROR: cannot load batch_ix {batch_ix} from {fname}")
            return None, None

    def load_counter_metadata(self):
        """ Returns 2 data frames as tuple for all saved training data with (counters, metadata).
        """
        try:
            fname = self.training_fname()
            counter_df = pd.read_hdf(fname, "counter")
            meta_df = pd.read_hdf(fname, "meta")
            return (counter_df, meta_df)
        except IOError:
            print(f"{env.timestr()} load data frame file ERROR: cannot load counter and metadata from {fname}")
            return None

    def save_data(self, data_df, counter_df):
        """ Saves a training data frame independent of base as well as the corresponding counters and metadata.
            This provides the means to shuffle across bases and load it in batches as needed.
        """
        fname = self.training_fname()
        bs = self.training_batch_size()
        batches = int(math.ceil(len(data_df) / bs))
        meta_df = pd.Series(
            {"batch_size": bs, "batches": batches, "samples": len(data_df),
                "features": self.fcls.data_mnemonic(), "targets": self.tcls.data_mnemonic()})
        print("{}: writing {} for {} samples as {} batches".format(
            datetime.now().strftime(Env.dt_format), fname, len(data_df), batches))
        meta_df.to_hdf(fname, "meta", mode="w")
        counter_df.to_hdf(fname, "counter", mode="w")
        for batch_ix in range(batches):
            sdf = data_df.iloc[batch_ix*bs:(batch_ix+1)*bs]
            sdf.to_hdf(fname, batch_ix, mode="w")


def prepare4keras(scaler, feature_df, target_df, target_class):
    samples = Bunch(
        data=feature_df.values,
        target=target_df.values,
        target_names=np.array(target_class.target_dict().keys()),
        feature_names=np.array(feature_df.keys()))
    if scaler is not None:
        samples.data = scaler.transform(samples.data)
    samples.target = keras.utils.to_categorical(
        samples.target, num_classes=samples.target_names.size())
    return samples.data, samples.target


class TrainingGenerator(keras.utils.Sequence):
    "Generates training data for Keras"
    def __init__(self, scaler, target_class: ct.Targets, feature_class: cf.Feature, shuffle=False):
        self.scaler = scaler
        self.shuffle = shuffle
        self.training = TrainingData(target_class, feature_class)
        (counter_df, meta_df) = self.training.load_counter_metadata()
        print(counter_df)

        self.fcls = feature_class
        self.tcls = target_class
        self.batches = meta_df["batches"]
        self.batch_size = meta_df["batch_size"]
        self.samples = meta_df["samples"]

    def __len__(self):
        "Denotes the number of batches per epoch"
        return self.batches

    def __getitem__(self, index):
        "Generate one batch of data"
        if index >= self.batches:
            print(f"TrainGenerator: index {index} > len {self.batches}")
            index %= self.batches
        batch_df = self.training.load_data(index)
        fdf = batch_df["features"]
        tdf = batch_df["targets"]
        return prepare4keras(self.scaler, fdf, tdf, self.tcls)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        print("TrainGenerator: on_epoch_end")
        if self.shuffle is True:
            np.random.shuffle(self.indices)


class ValidationGenerator(keras.utils.Sequence):
    'Generates validation data for Keras'
    def __init__(self, set_type, scaler, target_class: ct.Targets, feature_class: cf.Feature, shuffle=False):
        self.set_type = set_type
        self.scaler = scaler
        self.shuffle = shuffle
        self.fcls = feature_class
        self.tcls = target_class
        print(f"BaseGenerator: {Env.bases}")

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(Env.bases)

    def __getitem__(self, index):
        'Generate one batch of data'
        if index >= len(Env.bases):
            print(f"BaseGenerator: index {index} > len {len(Env.bases)}")
            index %= len(Env.bases)
        base = Env.bases[index]
        fdf = self.fcls.set_type_data(self.set_type, base)
        tdf = self.tcls.set_type_data(self.set_type, base)
        return prepare4keras(self.scaler, fdf, tdf, self.tcls)


if __name__ == "__main__":
    if True:
        td = TrainingData(ct.Targets, cof.F2cond24)
    else:
        td = TrainingData(ct.Targets, agf.F1agg110)
    if True:
        td.create_training_datasets()
    else:
        (cdf, mdf) = td.load_counter_metadata()
        print(cdf)
        print(mdf)
