import pandas as pd
# from queue import Queue
from datetime import timedelta
import math
# import numpy as np
# import sys
import env_config as env
from env_config import Env
import cached_crypto_data as ccd
import crypto_targets as ct
import crypto_features as cf
import condensed_features as cof

MANDATORY_STEPS = 2  # number of steps for the smallest class (in general BUY)
NA = "not assigned"
TRAIN = "training"
VAL = "validation"
TEST = "test"
LBL = {NA: 0, TRAIN: -1, VAL: -2, TEST: -3}
ActiveFeatures = cof.CondensedFeatures


class CryptoHistorySets:
    """Container class for targets and features of a currency pair
    timeblock is the time window in minutes that is analyzed equally for all bases to
    avoid label leakage. timeblocks are then distributed to train, validate and test to
    balance buy and sell signals.
    """
    # ! TODO: problem: history for feature calculation of interrupted train/test/val subsets
    # ! TODO: option: pre calc features and store them on disk, fetch set parts from disk

    def __init__(self, sets_config_fname):
        """Uses history data of baselist/USDT as history set to train and evaluate.

        training control:
        =================
        - df[common timeline index, base, target, tcount, buy_prob, sell_prob, hold_prob, use]
        - step within an epoch, every step-th class of a base is used
        - tcount is incremented with every training cycle usage
        - buy_prob, sell_prob, hold_prob are the class probabilities of the last evaluation
        - use is an earmark that this sample shall be used for the next training epoch/validation
        """
        self.bases = dict.fromkeys(Env.bases, None)  # all bases of history data available
        self.max_steps = dict.fromkeys(Env.bases)
        for base in self.max_steps:
            self.max_steps[base] = {ct.HOLD: 0, ct.BUY: 0, ct.SELL: 0, "max": 0}  # transaction count
        self.max_steps["total"] = 0
        self.timeblock = 4*7*24*60  # time window in minutes that is analyzed equally for all bases
        self.fixtic = None  # tic as fixpoint for timeblock
        self.analysis = pd.DataFrame(columns=["sym", "set_type", "start", "end", "tics",
                                              "buys", "sells", "avg_vol", "novol_count"])

        self.ctrl = dict()
        self.ctrl[TRAIN] = pd.DataFrame(columns=["base", "timestamp", "target", "use",
                                                 "hold_prob", "buy_prob", "sell_prob",
                                                 "step", "tcount"])
        self.ctrl[VAL] = pd.DataFrame(columns=["base", "timestamp", "target", "use",
                                               "hold_prob", "buy_prob", "sell_prob"])
        self.ctrl[TEST] = pd.DataFrame(columns=["base", "timestamp", "target", "use",
                                                "hold_prob", "buy_prob", "sell_prob"])
        self.last_base = None

        # for base in self.bases:
        #     merge_asset_dataframe(DATA_PATH, base)
        self.__load_sets_config(sets_config_fname)
        assert not self.analysis.empty, f"{env.timestr()}: missing sets config"
        cf.report_setsize(TRAIN, self.ctrl[TRAIN])
        cf.report_setsize(VAL, self.ctrl[VAL])
        cf.report_setsize(TEST, self.ctrl[TEST])
        # self.analyze_bases()

    def set_of_type(self, base, set_type):
        """ returns the base data of a specific type: TRAIN, VAL, TEST
            if the system is configured as having less than 16GB RAM then the last base data is released
        """
        try:
            base_df = self.ctrl[set_type].loc[(self.ctrl[set_type].base == base) &
                                              (self.ctrl[set_type].use.__eq__(True))]
            # print(f"{set_type} set with {len(base_df)} samples for {base}")
            return base_df
        except KeyError:
            print(f"no {self.set_type} set for {base}")
            pass

    def trainset_step(self, base, step):  # public
        """ returns the training subset of base for the next training step
        """
        try:
            hold_step = step % self.max_steps[base][ct.HOLD]
            buy_step = step % self.max_steps[base][ct.BUY]
            sell_step = step % self.max_steps[base][ct.SELL]
            base_df = self.ctrl[TRAIN].loc[(self.ctrl[TRAIN].base == base) &
                                           (((self.ctrl[TRAIN].target == ct.TARGETS[ct.HOLD]) &
                                             (self.ctrl[TRAIN].step == hold_step)) |
                                            ((self.ctrl[TRAIN].target == ct.TARGETS[ct.BUY]) &
                                             (self.ctrl[TRAIN].step == buy_step)) |
                                            ((self.ctrl[TRAIN].target == ct.TARGETS[ct.SELL]) &
                                             (self.ctrl[TRAIN].step == sell_step))) &
                                           (self.ctrl[TRAIN].use.__eq__(True))]
            # report_setsize(f"{base} {TRAIN} set step {step}", base_df)
            return base_df
        except KeyError:
            print(f"no {self.set_type} set for {base}")
            pass

    def __load_sets_config(self, config_fname):
        """ loads the configuration text file that defines the split into TRAIN, TEST, VAL
        """

        def use_settype_total():
            """Uses the set_type of "total" and apllies it to all sets with such timeblock.
            """
            cdf = self.analysis.set_index("end")
            cdf["set_type"] = cdf.loc[cdf.sym == "total"]["set_type"]
            # cdf["end"] = cdf.index  # is already doen by reset_index
            self.analysis = cdf.reset_index()

        try:
            self.analysis = pd.read_csv(config_fname, skipinitialspace=True, sep="\t")
        except IOError:
            print(f"pd.read_csv({config_fname}) IO error")
            return None
        # use_settype_total()
        # self.analysis.to_csv(config_fname, sep="\t", index=False)
        self.__prepare_training()

    def features_from_targets(self, df):
        if df.empty:
            raise cf.NoSubsetWarning("empty subset")
        base = df.at[df.index[0], "base"]
        minute_data = ccd.load_asset_dataframe(base, path=Env.data_path, limit=None)
        subset_df = cf.targets_to_features(minute_data, df)
        tf = ActiveFeatures(base, minute_dataframe=subset_df)
        return tf.calc_features_and_targets()

    def __samples_concat(self, target, to_be_added):
        if (target is None) or (target.empty):
            target = to_be_added
            # print("target empty --> target = to_be_added", target.head(), target.tail())
            return to_be_added
        if False:
            # debugging output
            elen = len(target)
            xdf = target.tail()
            if ("step" in xdf.columns):
                xdf = xdf[["target", "timestamp", "step"]]
            else:
                xdf = xdf[["target", "timestamp"]]
            print(f"target len: {elen}", xdf)
            ydf = to_be_added.head()
            if ("step" in ydf.columns):
                ydf = ydf[["target", "timestamp", "step"]]
            else:
                ydf = ydf[["target", "timestamp"]]
            print(f"time agg timeblock len: {len(to_be_added)}", ydf)
        target = pd.concat([target, to_be_added], sort=False)
        if False:
            # debugging output
            zdf = target.iloc[range(elen-5, elen+5)]
            elen = len(target)
            if ("step" in zdf.columns):
                zdf = zdf[["target", "timestamp", "step"]]
            else:
                zdf = zdf[["target", "timestamp"]]
            print(f"concat with new len {elen} result at interface: ", zdf)
        return target

    def __extract_set_type_targets(self, base, tf, set_type):
        """ Extracts the time ranges of set_type and the corresponding target and cose data from ohlcv data.
            Initializes an internal DataFrame with that data.
        """
        sym = env.sym_of_base(base)
        try:
            # print(f"extracting {set_type} for {sym}")
            dfcfg = self.analysis.loc[(self.analysis.set_type == set_type) &
                                      (self.analysis.sym == sym)]
        except KeyError:
            print(f"no {set_type} set for {sym}")
            return None
        dft = tf.minute_data
        extract = None
        for block in dfcfg.index:
            df = dft.loc[(dft.index >= dfcfg.at[block, "start"]) &
                         (dft.index <= dfcfg.at[block, "end"]), ["target", "close"]]
            df["timestamp"] = df.index
            df["base"] = base
            df["use"] = True
            df["buy_prob"] = float(0)
            df["sell_prob"] = float(0)
            df["hold_prob"] = float(0)
            if set_type == TRAIN:
                df["tcount"] = int(0)
                df["step"] = int(0)
            extract = self.__samples_concat(extract, df)
        self.ctrl[set_type] = self.__samples_concat(self.ctrl[set_type], extract)

    def __prepare_training(self):
        """Prepares training, validation and test sets with targets and admin info
        (see class description). These determine the samples per set_type and whether
        they are used in a step.

        It is assumed that load_sets_config was called and self.analysis contains the
        proper config file content.
        """

        for base in self.bases:
            tf = ActiveFeatures(base, path=Env.data_path)
            self.__extract_set_type_targets(base, tf, TRAIN)
            self.__extract_set_type_targets(base, tf, VAL)
            self.__extract_set_type_targets(base, tf, TEST)
            # del tf

    def register_probabilties(self, base, set_type, pred, target_df):
        """ Used by performance evaluation to register prediction probabilities.
            Those registered probabilities are evaluated by 'use_mistakes'.
        """
        df = self.ctrl[set_type]
        tdf = target_df
        # df.loc[df.index.isin(tdf.index) & (df.base == base), "hold_prob"] = pd.DataFrame(pred[:, ct.TARGETS[ct.HOLD]])
        # df.loc[df.index.isin(tdf.index) & (df.base == base), "buy_prob"] = pd.DataFrame(pred[:, ct.TARGETS[ct.BUY]])
        # df.loc[df.index.isin(tdf.index) & (df.base == base), "sell_prob"] = pd.DataFrame(pred[:, ct.TARGETS[ct.SELL]])
        df.loc[df.index.isin(tdf.index) & (df.base == base), ["hold_prob", "buy_prob", "sell_prob"]] = \
            pd.DataFrame(pred[:, [ct.TARGETS[ct.HOLD], ct.TARGETS[ct.BUY], ct.TARGETS[ct.SELL]]])
        if set_type == TRAIN:
            df.loc[tdf.index, "tcount"] = df.loc[tdf.index, "tcount"] + 1

    def use_mistakes(self, set_type):
        """ Currently unused but if used then by classify_keras!

            Sets the 'use' flag of all wrong classified samples to focus on mistakes.
        """
        df = self.ctrl[set_type]
        df["use"] = False
        df.loc[(df.target == ct.TARGETS[ct.HOLD]) &
               ((df.buy_prob >= df.hold_prob) | (df.sell_prob >= df.hold_prob)), "use"] = True
        df.loc[(df.target == ct.TARGETS[ct.BUY]) &
               ((df.hold_prob >= df.buy_prob) | (df.sell_prob >= df.buy_prob)), "use"] = True
        df.loc[(df.target == ct.TARGETS[ct.SELL]) &
               ((df.buy_prob >= df.sell_prob) | (df.hold_prob >= df.sell_prob)), "use"] = True
        return len(df[df.use.__eq__(True)])

    def label_steps(self):  # public
        """Each sample is assigned to a step. Consecutive samples are assigned to different steps.

        The smallest class has
        MANDATORY_STEPS, i.e. each sample is labeled
        with a step between 0 and MANDATORY_STEPS - 1.
        The larger classes have more step labels according to
        their ratio with the smallest class. This is determined per base,
        timeblock and target class.

        This sample distribution over steps shall balance the training of classes
        not bias the classifier too much in an early learning cycle.
        """
        self.max_steps["total"] = 0
        for base in self.bases:
            tdf = self.ctrl[TRAIN]
            tdf = tdf[tdf.base == base]
            self.max_steps[base] = {ct.HOLD: 0, ct.BUY: 0, ct.SELL: 0}
            holds = len(tdf[(tdf.target == ct.TARGETS[ct.HOLD]) & (tdf.use.__eq__(True))])
            sells = len(tdf[(tdf.target == ct.TARGETS[ct.SELL]) & (tdf.use.__eq__(True))])
            buys = len(tdf[(tdf.target == ct.TARGETS[ct.BUY]) & (tdf.use.__eq__(True))])
            all_use = len(tdf[(tdf.use.__eq__(True))])
            all_base = len(tdf)
            samples = holds + sells + buys
            # print(f"{base} buys:{buys} sells:{sells} holds:{holds} total:{samples} on {TRAIN}")
            if all_use != samples:
                print(f"samples {samples} != all use {all_use} as subset from all {base} {all_base}")
            min_step = min([buys, sells, holds])
            if min_step == 0:
                continue
            b_step = round(buys / min_step * MANDATORY_STEPS)
            s_step = round(sells / min_step * MANDATORY_STEPS)
            h_step = round(holds / min_step * MANDATORY_STEPS)
            bi = si = hi = 0
            tix = tdf.columns.get_loc("target")
            iix = tdf.columns.get_loc("step")
            for ix in range(len(tdf.index)):
                target = tdf.iat[ix, tix]
                if target == ct.TARGETS[ct.BUY]:
                    tdf.iat[ix, iix] = bi
                    bi = (bi + 1) % b_step
                elif target == ct.TARGETS[ct.SELL]:
                    tdf.iat[ix, iix] = si
                    si = (si + 1) % s_step
                elif target == ct.TARGETS[ct.HOLD]:
                    tdf.iat[ix, iix] = hi
                    hi = (hi + 1) % h_step
                else:  # hold
                    print(f"error: unexpected target {target}")
            self.ctrl[TRAIN].loc[(self.ctrl[TRAIN].base == base), "step"] = tdf.step
            self.max_steps[base][ct.BUY] = b_step
            self.max_steps[base][ct.SELL] = s_step
            self.max_steps[base][ct.HOLD] = h_step
            self.max_steps[base]["max"] = max([h_step, b_step, s_step])
            self.max_steps["total"] += self.max_steps[base]["max"]
        return self.max_steps["total"]

    def __label_check(self):
        """ Currently unused !

            Consistency check logging
        """
        print("label_check ==> maxsteps total:{}".format(self.max_steps["total"]))
        for base in self.bases:
            print("{} maxsteps of buy:{} sell:{} hold:{} max:{}".format(
                    base, self.max_steps[base][ct.BUY], self.max_steps[base][ct.SELL],
                    self.max_steps[base][ct.HOLD], self.max_steps[base]["max"]))
            holds = sells = buys = totals = 0
            for step in range(self.max_steps[base]["max"]):
                df = self.trainset_step(base, step)
                hc = len(df[df.target == ct.TARGETS[ct.HOLD]])
                sc = len(df[df.target == ct.TARGETS[ct.SELL]])
                bc = len(df[df.target == ct.TARGETS[ct.BUY]])
                tc = hc + sc + bc
                print(f"buy {bc} sell {sc} hold {hc} total {tc} on label_check {step}")
                if step < self.max_steps[base]["max"]:
                    holds += hc
                    sells += sc
                    buys += bc
                    totals += tc
            df = self.set_of_type(base, TRAIN)
            hc = len(df[df.target == ct.TARGETS[ct.HOLD]])
            sc = len(df[df.target == ct.TARGETS[ct.SELL]])
            bc = len(df[df.target == ct.TARGETS[ct.BUY]])
            nc = len(df)
            tc = hc + sc + bc
            print(f"label check set: buy {bc} sell {sc} hold {hc} total {tc} whole set{nc}")

    def __analyze_bases(self):
        """ Currently unused !

            Analyses the baselist and creates a config file of sets split into equal timeblock
        length. Additionally an artificial "total" set is created with the sum of all bases for
        each timeblock.
        """

        def first_week_ts(tf):
            df = tf.minute_data
            assert df is not None
            first_tic = df.index[0].to_pydatetime()
            last_tic = df.index[len(df.index)-1].to_pydatetime()
            if self.fixtic is None:
                self.fixtic = last_tic
            fullweeks = math.ceil((self.fixtic - first_tic) / timedelta(minutes=self.timeblock))
            assert fullweeks > 0, \
                f"{len(df)} {len(df.index)} {first_tic} {last_tic} {fullweeks}"
            wsts = self.fixtic - timedelta(minutes=fullweeks*self.timeblock-1)
            wets = wsts + timedelta(minutes=self.timeblock-1)
            return (wsts, wets)

        def next_week_ts(tf, wets):
            wsts = wets + timedelta(minutes=1)
            wets = wsts + timedelta(minutes=self.timeblock-1)
            df = tf.minute_data
            assert df is not None
            last_tic = df.index[len(df.index)-1].to_pydatetime()
            if last_tic < wsts:
                return (None, None)
            else:
                return (wsts, wets)  # first and last tic of a block

        def analyze_timeblock(tf, wsts, wets, sym):
            dfm = tf.minute_data
            assert dfm is not None
            dfm = dfm.loc[(dfm.index >= wsts) & (dfm.index <= wets), ["target", "volume"]]
            buys = len(dfm.loc[dfm.target == ct.TARGETS[ct.BUY]])
            sells = len(dfm.loc[dfm.target == ct.TARGETS[ct.SELL]])
            vcount = len(dfm)
            avgvol = int(dfm["volume"].mean())
            novol = len(dfm.loc[dfm.volume <= 0])
            lastix = len(self.analysis)
            self.analysis.loc[lastix] = [sym, NA, wsts, wets, vcount, buys, sells, avgvol, novol]

        blocks = set()
        for base in self.bases:
            sym = base + "_usdt"
            tf = ActiveFeatures(base, path=Env.data_path)
            (wsts, wets) = first_week_ts(tf)
            while wets is not None:
                blocks.add(wets)
                analyze_timeblock(tf, wsts, wets, sym)
                (wsts, wets) = next_week_ts(tf, wets)
        blocklist = list(blocks)
        blocklist.sort()
        for wets in blocklist:
            df = self.analysis.loc[self.analysis.end == wets]
            buys = int(df["buys"].sum())
            sells = int(df["sells"].sum())
            avgvol = int(df["avg_vol"].mean())
            novol = int(df["novol_count"].sum())
            vcount = int(df["tics"].sum())
            wsts = wets - timedelta(minutes=self.timeblock)
            lastix = len(self.analysis)
            sym = "total"
            self.analysis.loc[lastix] = [sym, NA, wsts, wets, vcount, buys, sells, avgvol, novol]
        cfname = env.sets_config_fname()
        self.analysis.to_csv(cfname, sep="\t", index=False)


if __name__ == "__main__":
    env.test_mode()
    hs = CryptoHistorySets(env.sets_config_fname())
    for set_type in [TRAIN, VAL, TEST]:
        for base in hs.bases:
            bix = list(hs.bases.keys()).index(base)
            df = hs.set_of_type(base, set_type)
            tfv = hs.features_from_targets(df)
            descr = "{} {} {} set step {}: {}".format(env.timestr(), base, set_type,
                                                      bix, cf.str_setsize(tfv))
            print(descr)
