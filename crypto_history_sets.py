import pandas as pd
# from queue import Queue
from datetime import timedelta
import math
# import numpy as np
# import sys
import env_config as env
from env_config import Env
# import cached_crypto_data as ccd
import crypto_targets_features as ctf
from crypto_targets_features import HOLD, BUY, SELL, NA, TRAIN, VAL, TEST, TARGETS, MANDATORY_STEPS


class CryptoHistorySets:
    """Container class for targets and features of a currency pair
    timeblock is the time window in minutes that is analyzed equally for all bases to
    avoid label leakage. timeblocks are then distributed to train, validate and test to
    balance buy and sell signals.
    """

    def __init__(self, sets_config_fname):
        """Uses history data of baselist/USDT as history set to train and evaluate.

        training control:
        =================
        - df[common timeline index, base, target, tcount, buy_prob, sell_prob, hold_prob, use]
        - step within an epoch, every step-th class of a sym is used
        - tcount is incremented with every training cycle usage
        - buy_prob, sell_prob, hold_prob are the class probabilities of the last evaluation
        - use is an earmark that this sample shall be used for the next training epoch/validation
        """
        self.bases = dict.fromkeys(Env.bases, None)  # all bases of history data available
        self.max_steps = dict.fromkeys(Env.bases)
        for base in self.max_steps:
            self.max_steps[base] = {HOLD: 0, BUY: 0, SELL: 0, "max": 0}  # transaction count
        self.max_steps["total"] = 0
        self.timeblock = 4*7*24*60  # time window in minutes that is analyzed equally for all bases
        self.fixtic = None  # tic as fixpoint for timeblock
        self.analysis = pd.DataFrame(columns=["sym", "set_type", "start", "end", "tics",
                                              "buys", "sells", "avg_vol", "novol_count"])

        self.ctrl = dict()
        self.ctrl[TRAIN] = pd.DataFrame(columns=["sym", "timestamp", "target", "use",
                                                 "hold_prob", "buy_prob", "sell_prob",
                                                 "step", "tcount"])
        self.ctrl[VAL] = pd.DataFrame(columns=["sym", "timestamp", "target", "use",
                                               "hold_prob", "buy_prob", "sell_prob"])
        self.ctrl[TEST] = pd.DataFrame(columns=["sym", "timestamp", "target", "use",
                                                "hold_prob", "buy_prob", "sell_prob"])
        self.last_base = None

        # for base in self.bases:
        #     merge_asset_dataframe(DATA_PATH, base)
        self.__load_sets_config(sets_config_fname)
        assert not self.analysis.empty, f"{env.timestr()}: missing sets config"
        ctf.report_setsize(TRAIN, self.ctrl[TRAIN])
        ctf.report_setsize(VAL, self.ctrl[VAL])
        ctf.report_setsize(TEST, self.ctrl[TEST])
        # self.analyze_bases()

    def set_of_type(self, base, set_type):
        """ returns the base data of a specific type: TRAIN, VAL, TEST
            if the system is configured as having less than 16GB RAM then the last base data is released
        """
        sym = env.sym_of_base(base)
        if Env.smaller_16gb_ram and (self.last_base != base):
            self.__release_features_of_base(self.last_base)
        try:
            base_df = self.ctrl[set_type].loc[(self.ctrl[set_type].sym == sym) &
                                              (self.ctrl[set_type].use.__eq__(True))]
            # print(f"{set_type} set with {len(base_df)} samples for {sym}")
            return base_df
        except KeyError:
            print(f"no {self.set_type} set for {sym}")
            pass

    def trainset_step(self, base, step):  # public
        """ returns the training subset of base for the next training step
        """
        sym = env.sym_of_base(base)
        if Env.smaller_16gb_ram and (self.last_base != base):
            self.__release_features_of_base(self.last_base)
        try:
            hold_step = step % self.max_steps[base][HOLD]
            buy_step = step % self.max_steps[base][BUY]
            sell_step = step % self.max_steps[base][SELL]
            base_df = self.ctrl[TRAIN].loc[(self.ctrl[TRAIN].sym == sym) &
                                           (((self.ctrl[TRAIN].target == TARGETS[HOLD]) &
                                             (self.ctrl[TRAIN].step == hold_step)) |
                                            ((self.ctrl[TRAIN].target == TARGETS[BUY]) &
                                             (self.ctrl[TRAIN].step == buy_step)) |
                                            ((self.ctrl[TRAIN].target == TARGETS[SELL]) &
                                             (self.ctrl[TRAIN].step == sell_step))) &
                                           (self.ctrl[TRAIN].use.__eq__(True))]
            # report_setsize(f"{sym} {TRAIN} set step {step}", base_df)
            return base_df
        except KeyError:
            print(f"no {self.set_type} set for {sym}")
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

    def features_from_targets(self, df, base, set_type, step):
        if df.empty:
            raise ctf.NoSubsetWarning("empty {} subset for {}".format(set_type, base))
        sym = df.at[df.index[0], "sym"]
        df_base = env.base_of_sym(sym)
        if base != df_base:
            raise ValueError(f"features_from_targets: base(df)={df_base} != base={base}")
        tfv = self.get_targets_features_of_base(base)
        try:
            subset_df = ctf.targets_to_features(tfv, df)
        except ctf.NoSubsetWarning as msg:
            print("features_from_targets  {} {} set step {}: {}".format(
                        base, set_type, step, msg))
            raise
        return subset_df

    def __samples_concat(self, target, to_be_added):
        if target.empty:
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
            df["sym"] = sym
            df["use"] = True
            df["buy_prob"] = float(0)
            df["sell_prob"] = float(0)
            df["hold_prob"] = float(0)
            if set_type == TRAIN:
                df["tcount"] = int(0)
                df["step"] = int(0)
            if extract is None:
                extract = df
            else:
                extract = self.__samples_concat(extract, df)
        return extract

    def __prepare_training(self):
        """Prepares training, validation and test sets with targets and admin info
        (see class description). These determine the samples per set_type and whether
        they are used in a step.

        It is assumed that load_sets_config was called and self.analysis contains the
        proper config file content.
        """

        for base in self.bases:
            tf = ctf.TargetsFeatures(base)
            try:
                tf.load_classifier_features()
            except env.MissingHistoryData:
                continue
            self.bases[base] = tf
            tfv = tf.vec
            tdf = self.__extract_set_type_targets(base, tf, TRAIN)
            tdf = tdf[tdf.index.isin(tfv.index)]
            self.ctrl[TRAIN] = self.__samples_concat(self.ctrl[TRAIN], tdf)
            vdf = self.__extract_set_type_targets(base, tf, VAL)
            vdf = vdf[vdf.index.isin(tfv.index)]
            self.ctrl[VAL] = self.__samples_concat(self.ctrl[VAL], vdf)
            tstdf = self.__extract_set_type_targets(base, tf, TEST)
            tstdf = tstdf[tstdf.index.isin(tfv.index)]
            self.ctrl[TEST] = self.__samples_concat(self.ctrl[TEST], tstdf)

    def get_targets_features_of_base(self, base):
        if base not in self.bases:
            raise KeyError()
        tf = self.bases[base]
        if tf is None:
            tf = ctf.TargetsFeatures(base)
            tf.load_classifier_features()
        if tf is not None:
            if tf.vec is None:
                try:
                    tf.calc_features_and_targets(None)
                except env.MissingHistoryData as msg:
                    print(f"get_targets_features_of_base {base}: {msg}")
            return tf.vec
        return None

    def __release_features_of_base(self, base):
        """ setting the only refernce to the base data to None releases the data from RAM through garbage collection
        """
        if base in self.bases:
            tf = self.bases[base]
            if tf is not None:
                tf.vec = None

    def register_probabilties(self, base, set_type, pred, target_df):
        df = self.ctrl[set_type]
        tdf = target_df
        sym = env.sym_of_base(base)
        # df.loc[df.index.isin(tdf.index) & (df.sym == sym), "hold_prob"] = pd.DataFrame(pred[:, TARGETS[HOLD]])
        # df.loc[df.index.isin(tdf.index) & (df.sym == sym), "buy_prob"] = pd.DataFrame(pred[:, TARGETS[BUY]])
        # df.loc[df.index.isin(tdf.index) & (df.sym == sym), "sell_prob"] = pd.DataFrame(pred[:, TARGETS[SELL]])
        df.loc[df.index.isin(tdf.index) & (df.sym == sym), ["hold_prob", "buy_prob", "sell_prob"]] = \
            pd.DataFrame(pred[:, [TARGETS[HOLD], TARGETS[BUY], TARGETS[SELL]]])
        if set_type == TRAIN:
            df.loc[tdf.index, "tcount"] = df.loc[tdf.index, "tcount"] + 1

    def __use_mistakes(self, set_type):
        """ Sets the 'use' flag of all wrong classified samples to focus on mistakes.
        """
        df = self.ctrl[set_type]
        df["use"] = False
        df.loc[(df.target == TARGETS[HOLD]) &
               ((df.buy_prob >= df.hold_prob) | (df.sell_prob >= df.hold_prob)), "use"] = True
        df.loc[(df.target == TARGETS[BUY]) &
               ((df.hold_prob >= df.buy_prob) | (df.sell_prob >= df.buy_prob)), "use"] = True
        df.loc[(df.target == TARGETS[SELL]) &
               ((df.buy_prob >= df.sell_prob) | (df.hold_prob >= df.sell_prob)), "use"] = True
        return len(df[df.use.__eq__(True)])

    def __base_label_check(self, base):
        print("{} maxsteps of buy:{} sell:{} hold:{} max:{}".format(
                base, self.max_steps[base][BUY], self.max_steps[base][SELL],
                self.max_steps[base][HOLD], self.max_steps[base]["max"]))
        holds = sells = buys = totals = 0
        for step in range(self.max_steps[base]["max"]):
            df = self.trainset_step(base, step)
            hc = len(df[df.target == TARGETS[HOLD]])
            sc = len(df[df.target == TARGETS[SELL]])
            bc = len(df[df.target == TARGETS[BUY]])
            tc = hc + sc + bc
            print(f"buy {bc} sell {sc} hold {hc} total {tc} on label_check {step}")
            if step < self.max_steps[base]["max"]:
                holds += hc
                sells += sc
                buys += bc
                totals += tc
        df = self.set_of_type(base, TRAIN)
        hc = len(df[df.target == TARGETS[HOLD]])
        sc = len(df[df.target == TARGETS[SELL]])
        bc = len(df[df.target == TARGETS[BUY]])
        nc = len(df)
        tc = hc + sc + bc
        print(f"label check set: buy {bc} sell {sc} hold {hc} total {tc} whole set{nc}")

    def __label_check(self):
        print("label_check ==> maxsteps total:{}".format(self.max_steps["total"]))
        for base in self.bases:
            self.__base_label_check(base)

    def label_steps(self):  # public
        """Each sample is assigned to a step. The smallest class has
        MANDATORY_STEPS, i.e. each sample is labeled
        with a step between 0 and MANDATORY_STEPS - 1.
        The larger classes have more step labels according to
        their ratio with the smallest class. This is determined per sym,
        timeblock and target class.

        This sample distribution over steps shall balance the training of classes
        not bias the classifier too much in an early learning cycle.
        """
        self.max_steps["total"] = 0
        for base in self.bases:
            sym = env.sym_of_base(base)
            tdf = self.ctrl[TRAIN]
            tdf = tdf[tdf.sym == sym]
            self.max_steps[base] = {HOLD: 0, BUY: 0, SELL: 0}
            holds = len(tdf[(tdf.target == TARGETS[HOLD]) & (tdf.use.__eq__(True))])
            sells = len(tdf[(tdf.target == TARGETS[SELL]) & (tdf.use.__eq__(True))])
            buys = len(tdf[(tdf.target == TARGETS[BUY]) & (tdf.use.__eq__(True))])
            all_use = len(tdf[(tdf.use.__eq__(True))])
            all_sym = len(tdf)
            samples = holds + sells + buys
            # print(f"{sym} buys:{buys} sells:{sells} holds:{holds} total:{samples} on {TRAIN}")
            if all_use != samples:
                print(f"samples {samples} != all use {all_use} as subset from all {sym} {all_sym}")
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
                if target == TARGETS[BUY]:
                    tdf.iat[ix, iix] = bi
                    bi = (bi + 1) % b_step
                elif target == TARGETS[SELL]:
                    tdf.iat[ix, iix] = si
                    si = (si + 1) % s_step
                elif target == TARGETS[HOLD]:
                    tdf.iat[ix, iix] = hi
                    hi = (hi + 1) % h_step
                else:  # hold
                    print(f"error: unexpected target {target}")
            self.ctrl[TRAIN].loc[(self.ctrl[TRAIN].sym == sym), "step"] = tdf.step
            self.max_steps[base][BUY] = b_step
            self.max_steps[base][SELL] = s_step
            self.max_steps[base][HOLD] = h_step
            self.max_steps[base]["max"] = max([h_step, b_step, s_step])
            self.max_steps["total"] += self.max_steps[base]["max"]
        return self.max_steps["total"]

    def __analyze_bases(self):
        """Analyses the baselist and creates a config file of sets split into equal timeblock
        length. Additionally an artificial "total" set is created with the sum of all bases for
        each timeblock.
        """

        def first_week_ts(tf):
            df = tf.vec
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
            df = tf.vec
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
            buys = len(dfm.loc[dfm.target == TARGETS[BUY]])
            sells = len(dfm.loc[dfm.target == TARGETS[SELL]])
            vcount = len(dfm)
            avgvol = int(dfm["volume"].mean())
            novol = len(dfm.loc[dfm.volume <= 0])
            lastix = len(self.analysis)
            self.analysis.loc[lastix] = [sym, NA, wsts, wets, vcount, buys, sells, avgvol, novol]

        blocks = set()
        for base in self.bases:
            sym = base + "_usdt"
            tf = ctf.TargetsFeatures(base)
            tf.load_classifier_features()
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
