import os
# import timeit
import numpy as np
import pandas as pd
import logging
# import cached_crypto_data as ccd
import crypto_targets as ct
# import adaptation_data as ad
import tensorflow as tf

logger = logging.getLogger(__name__)
# tf.config.experimental_run_functions_eagerly(True)


class PerfMatrix:
    """Evaluates the performance across a range of buy/sell thresholds
    """

    def __init__(self):
        self.PERF = tf.constant(0, dtype=tf.int32)
        self.COUNT = tf.constant(1, dtype=tf.int32)
        self.BUY_IX = self.BT_IX = tf.constant(2, dtype=tf.int32)
        self.BUY_PRICE = self.ST_IX = tf.constant(3, dtype=tf.int32)
        track = [self.PERF, self.COUNT, self.BUY_IX, self.BUY_PRICE]
        self.btl = tf.constant([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=tf.float64)  # buy thresholds
        self.stl = tf.constant([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=tf.float64)  # buy thresholds
        self.perf = tf.Variable(tf.zeros((len(self.btl), len(self.stl), len(track)), dtype=tf.float64))
        # self.perf = overall performance array
        self.conf = tf.Variable(tf.zeros((len(ct.TARGETS), len(ct.TARGETS)), dtype=tf.float64))

        self.close_arr = None
        self.target_arr = None
        self.pred_np = None

    def sell_transaction(self, sample, btix, stix, buy_price):
        # logger.debug(f"sample {sample}, btix {btix}, stix {stix}, buy_price {buy_price}")
        transaction_perf = (self.close_arr[sample] * (1 - ct.FEE) - buy_price * (1 + ct.FEE)) / buy_price * (1 + ct.FEE)
        self.perf = tf.tensor_scatter_nd_update(self.perf, [[btix, stix, self.BUY_PRICE]], [0])
        self.perf = tf.tensor_scatter_nd_add(
            self.perf, [[btix, stix, self.PERF], [btix, stix, self.COUNT]], [transaction_perf, 1])

    def sell_ge_threshold(self, sample, stix):
        # logger.debug(f"sample {sample}, stix {stix}")
        for btix in range(len(self.btl)):
            buy_price = self.perf[btix, stix, self.BUY_PRICE]
            tf.cond((buy_price > 0),
                    lambda: self.sell_transaction(sample, btix, stix, buy_price),
                    lambda: None)

    def sell(self, sample):
        """ process sell signal without confusion matrix update
        """
        # logger.debug(f"sample {sample}")
        pred_sell = self.pred_np[sample, ct.TARGETS[ct.SELL]]
        for stix in range(len(self.stl)):
            tf.cond((pred_sell >= self.stl[stix]),
                    lambda: self.sell_ge_threshold(sample, stix),
                    lambda: None)
        self.conf = tf.tensor_scatter_nd_add(self.conf, [[self.target_arr[sample], ct.TARGETS[ct.SELL]]], [1])

    def hold(self, sample):
        # logger.debug(f"sample {sample}")
        self.conf = tf.tensor_scatter_nd_add(self.conf, [[self.target_arr[sample], ct.TARGETS[ct.HOLD]]], [1])

    def sell_ge_buy(self, sample, pred_sell, pred_hold):
        # logger.debug(f"pred_sell {pred_sell}, pred_hold: {pred_hold}, sample: {sample}")
        tf.cond((pred_sell >= pred_hold),
                lambda: self.sell(sample),
                lambda: self.hold(sample))

    def revert_buy_transaction(self, btix, stix):
        # logger.debug(f"btix {btix}, stix {stix}")
        self.perf = tf.tensor_scatter_nd_update(self.perf, [[btix, stix, self.BUY_PRICE]], [0])

    def check_open_buy(self, last_sample, btix, stix, buy_price):
        # logger.debug(f"last_sample {last_sample}, btix {btix}, stix {stix}, buy_price {buy_price}")
        tf.cond((self.perf[btix, stix, self.BUY_IX] == last_sample),
                lambda: self.revert_buy_transaction(btix, stix),
                lambda: self.sell_transaction(last_sample, btix, stix, buy_price))

    def buy_transaction(self, sample, btix, stix):
        # first buy of a possible sequence of multiple buys before sell
        # logger.debug(f"sample {sample}, btix {btix}, stix {stix}")
        self.perf = tf.tensor_scatter_nd_update(
            self.perf, [[btix, stix, self.BUY_PRICE], [btix, stix, self.BUY_IX]], [self.close_arr[sample], sample])

    def buy_ge_threshold(self, sample, btix):
        # logger.debug(f"sample {sample}, btix {btix}")
        for stix in range(len(self.stl)):
            tf.cond((self.perf[btix, stix, self.BUY_PRICE] == 0),
                    lambda: self.buy_transaction(sample,
                    btix, stix),
                    lambda: None)

    def buy(self, sample):
        """ process sell signal without confusion matrix update
        """
        # logger.debug(f"sample {sample}")
        pred_buy = self.pred_np[sample, ct.TARGETS[ct.BUY]]
        for btix in range(len(self.btl)):
            tf.cond((pred_buy >= self.btl[btix]),
                    lambda: self.buy_ge_threshold(sample, btix),
                    lambda: None)
        self.conf = tf.tensor_scatter_nd_add(self.conf, [[self.target_arr[sample], ct.TARGETS[ct.BUY]]], [1])

    def sell_lt_buy(self, sample, pred_buy, pred_hold):
        # logger.debug(f"pred_buy {pred_buy}, pred_hold: {pred_hold}, sample: {sample}")
        tf.cond((pred_buy >= pred_hold),
                lambda: self.buy(sample),
                lambda: self.hold(sample))

    @tf.function
    def _assess_prediction_np(self):
        """ Assesses the performance with different buy/sell thresholds. This method can be called multiple times,
            which accumulates the performance results in the class attributes 'perf' and 'conf'

            - pred_np is a tensor with numpy data returning the predictions per class
                with class index == ct.TARGETS[BUY|SELL|HOLD]
            - close_df is a data frame with a 'close' column containing close prices
            - target_df is a data frame with a 'target' column containing the target class index
        """
        pred_cnt, _ = self.pred_np.shape
        for sample in range(pred_cnt):
            pred_sell = self.pred_np[sample, ct.TARGETS[ct.SELL]]
            pred_buy = self.pred_np[sample, ct.TARGETS[ct.BUY]]
            pred_hold = self.pred_np[sample, ct.TARGETS[ct.HOLD]]
            tf.cond((pred_sell >= pred_buy),
                    lambda: self.sell_ge_buy(sample, pred_sell, pred_hold),
                    lambda: self.sell_lt_buy(sample, pred_buy, pred_hold))

        for btix in range(len(self.btl)):
            for stix in range(len(self.stl)):
                buy_price = self.perf[btix, stix, self.BUY_PRICE]
                tf.cond((buy_price > 0),
                        lambda: self.check_open_buy(pred_cnt-1, btix, stix, buy_price),
                        lambda: None)

        for btix in range(len(self.btl)):
            for stix in range(len(self.stl)):
                self.perf = tf.tensor_scatter_nd_update(self.perf, [[btix, stix, self.BT_IX]], [self.btl[btix]])
                self.perf = tf.tensor_scatter_nd_update(self.perf, [[btix, stix, self.ST_IX]], [self.stl[stix]])

    def assess_prediction_np(self, pred_np, close_arr, target_arr, perf, conf):
        self.close_arr = tf.constant(close_arr)
        self.target_arr = tf.constant(target_arr)
        self.pred_np = tf.constant(pred_np)
        self._assess_prediction_np()
        return self.perf, self.conf

    def best_np(self, perf):
        bp = -999999.9
        bc = bbt = bst = 0
        btlen, stlen, _ = perf.shape
        for btix in range(btlen):
            for stix in range(stlen):
                if perf[btix, stix, self.PERF] > bp:
                    (bp, bc, bbt, bst) = \
                        (float(perf[btix, stix, self.PERF]), int(perf[btix, stix, self.COUNT]),
                         float(perf[btix, stix, self.BT_IX]), float(perf[btix, stix, self.ST_IX]))
        return (bp, bbt, bst, int(bc))

    def _prep_perf_mat(self, perf_np):
        btlen, stlen, _ = perf_np.shape
        btl = [float(perf_np[btix, 0, self.BT_IX]) for btix in range(btlen)]
        stl = [float(perf_np[0, stix, self.ST_IX]) for stix in range(stlen)]
        perf_mat = pd.DataFrame(index=btl, columns=pd.MultiIndex.from_product([stl, ["perf", "count"]]))
        perf_mat.index.rename("btix", inplace=True)
        perf_mat.columns.rename(["stix", "kpi"], inplace=True)
        for btix in range(btlen):
            for stix in range(stlen):
                perf_mat.loc[btl[btix], (stl[stix], "perf")] = "{:>5.0%}".format(float(perf_np[btix, stix, self.PERF]))
                perf_mat.loc[btl[btix], (stl[stix], "count")] = int(perf_np[btix, stix, self.COUNT])
        return perf_mat

    def report_assessment_np(self, perf, conf, name):
        conf_mat = pd.DataFrame(index=ct.TARGETS.keys(), columns=ct.TARGETS.keys(), dtype=int)
        conf_mat.index.rename("target", inplace=True)
        conf_mat.columns.rename("actual", inplace=True)
        for target_label in ct.TARGETS:
            for actual_label in ct.TARGETS:
                conf_mat.loc[target_label, actual_label] = \
                    int(conf[ct.TARGETS[target_label], ct.TARGETS[actual_label]])
        # logger.info(f"confusion matrix np: \n{conf}")
        logger.info(f"{name} confusion matrix overall: \n{conf_mat}")
        (bp, bbt, bst, bc) = self.best_np(perf)
        logger. info(f"best {name} performance overall {bp:>5.0%} at bt/st {bbt}/{bst} with {bc} transactions")
        # logger.info(f"performance matrix np: \n{perf}")
        logger.info(f"{name} performance matrix overall: \n{self._prep_perf_mat(perf)}")


@tf.function
def forward(x):
    W = np.ones((2, 2))  # overall performace array
    b = np.zeros((2))  # overall performace array
    return W * x + b


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# tf.config.set_visible_devices([], 'GPU')
# tf.config.experimental.set_visible_devices([], 'GPU')

if __name__ == "__main__":
    df = pd.DataFrame(
        data={"close": [100,  90, 100,  99,  90,  91, 120, 121, 130, 129, 110, 111, 100,  99, 110, 120, 125.0],
              "target": [00,   0,   1,   0,   0,   2,   1,   0,   0,   0,   2,   1,   0,   0,  1,    1,   1],
              ct.HOLD: [0.0, 0.3, 0.4, 0.7, 0.3, 0.4, 0.4, 0.7, 0.3, 0.5, 0.2, 0.4, 0.2, 0.6, 0.1, 0.0, 0.0],
              ct.BUY:  [0.0, 0.1, 0.6, 0.1, 0.0, 0.0, 0.6, 0.0, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9, 0.8],
              ct.SELL: [0.0, 0.6, 0.0, 0.0, 0.7, 0.2, 0.0, 0.2, 0.0, 0.0, 0.8, 0.3, 0.4, 0.2, 0.1, 0.1, 0.2]},
        columns=["close", "target", ct.HOLD, ct.BUY, ct.SELL],  # , "sell_ix", "sell_price", "perf"],
        index=pd.date_range('2012-10-08 18:15:05', periods=17, freq='T'))
    # print(f"inital \n {df.head(18)}")
    pred_df = df.loc[:, [ct.HOLD, ct.BUY, ct.SELL]]
    # print(f"inital \n {pred_df.head(18)}")

    pm = PerfMatrix()
    (perf, conf) = pm.assess_prediction_np(
        pred_df.values, df["close"].values, df["target"].values, pm.perf, pm.conf)

    pm.report_assessment_np(perf, conf, "test")
    print(perf, conf)

    # W = tf.Variable(tf.ones(shape=(2, 2)), name="W")
    # b = tf.Variable(tf.zeros(shape=(2)), name="b")

    # out_a = forward([1, 0])
    # print(out_a)
