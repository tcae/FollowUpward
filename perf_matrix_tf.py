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
tf.config.experimental_run_functions_eagerly(True)


def _process_sell(close_arr, sample, pred_sell, perf, btl, stl):
    """ process sell signal without confusion matrix update
    """
    (PERF, COUNT, BUY_IX, BUY_PRICE) = (ix for ix in range(4))
    for st in range(len(stl)):
        if pred_sell >= stl[st]:
            for bt in range(len(btl)):
                buy_price = perf[bt, st, BUY_PRICE]
                if buy_price > 0:
                    # add here the transaction tracking shall be inserted
                    transaction_perf = \
                        (close_arr[sample] * (1 - ct.FEE) - buy_price * (1 + ct.FEE)) \
                        / buy_price * (1 + ct.FEE)
                    perf = perf.scatter_nd_update([[bt, st, BUY_PRICE]], [0])
                    perf = perf.scatter_nd_add([[bt, st, PERF], [bt, st, COUNT]], [transaction_perf, 1])
    return perf


def _process_buy(close_arr, sample, pred_buy, perf, btl, stl):
    """ process sell signal without confusion matrix update
    """
    (PERF, COUNT, BUY_IX, BUY_PRICE) = (ix for ix in range(4))
    for bt in range(len(btl)):
        if pred_buy >= btl[bt]:
            for st in range(len(stl)):
                if perf[bt, st, BUY_PRICE] == 0:
                    # first buy of a possible sequence of multiple buys before sell
                    perf = perf.scatter_nd_update(
                        [[bt, st, BUY_PRICE], [bt, st, BUY_IX]], [close_arr[sample], sample])
    return perf


def _close_open_transactions_np(close_arr, last_ix, perf, btl, stl):
    """ only corrects the performance calculation but not the confusion matrix
    """
    (PERF, COUNT, BUY_IX, BUY_PRICE) = (ix for ix in range(4))
    for bt in range(len(btl)):
        for st in range(len(stl)):
            buy_price = perf[bt, st, BUY_PRICE]
            if buy_price > 0:
                if perf[bt, st, BUY_IX] == last_ix:
                    # last transaction was opened by last sample --> revert transaction
                    perf = perf.scatter_nd_update([[bt, st, BUY_PRICE]], [0])
                else:
                    # last transaction was opened before last sample --> close with sell
                    # add here the transaction tracking
                    transaction_perf = \
                        (close_arr[-1] * (1 - ct.FEE) - buy_price * (1 + ct.FEE)) / buy_price * (1 + ct.FEE)
                    perf = perf.scatter_nd_update([[bt, st, BUY_PRICE]], [0])
                    perf = perf.scatter_nd_add([[bt, st, PERF], [bt, st, COUNT]], [transaction_perf, 1])
    return perf


@tf.function
def assess_prediction_np(pred_np, close_arr, target_arr):
    """ Assesses the performance with different buy/sell thresholds. This method can be called multiple times,
        which accumulates the performance results in the class attributes 'perf' and 'conf'

        - pred_np is a tensor with numpy data returning the predictions per class
            with class index == ct.TARGETS[BUY|SELL|HOLD]
        - close_df is a data frame with a 'close' column containing close prices
        - target_df is a data frame with a 'target' column containing the target class index
    """
    elems = 4
    (PERF, COUNT, BUY_IX, BUY_PRICE) = (ix for ix in range(elems))
    btl = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # buy thresholds
    stl = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # sell thresholds
    perf = tf.Variable(tf.zeros((len(btl), len(stl), elems), dtype=tf.float32))  # overall performace array
    conf = tf.Variable(tf.zeros((len(ct.TARGETS), len(ct.TARGETS)), dtype=tf.float32))
    # conf = (target, actual) confusion matrix

    if (pred_np is None) or (close_arr is None) or (target_arr is None):
        return (perf, conf)

    pred_cnt = len(pred_np)
    BUY = ct.TARGETS[ct.BUY]
    SELL = ct.TARGETS[ct.SELL]
    HOLD = ct.TARGETS[ct.HOLD]
    for sample in range(pred_cnt):
        pred_sell = pred_np[sample, SELL]
        pred_buy = pred_np[sample, BUY]
        pred_hold = pred_np[sample, HOLD]
        if pred_sell >= pred_buy:
            if pred_sell >= pred_hold:
                # SELL
                perf = _process_sell(close_arr, sample, pred_sell, perf, btl, stl)
                conf = conf.scatter_nd_add([[target_arr[sample], SELL]], [1])
            else:
                # HOLD
                conf = conf.scatter_nd_add([[target_arr[sample], HOLD]], [1])
        else:  # SELL < HOLD
            if pred_buy >= pred_hold:
                # BUY
                perf = _process_buy(close_arr, sample, pred_buy, perf, btl, stl)
                conf = conf.scatter_nd_add([[target_arr[sample], BUY]], [1])
            else:
                # HOLD
                conf = conf.scatter_nd_add([[target_arr[sample], HOLD]], [1])
    perf = _close_open_transactions_np(close_arr, pred_cnt-1, perf, btl, stl)
    # reuse BUY_IX, BUY_PRICE fields for bt and st
    BT_IX = BUY_IX
    ST_IX = BUY_PRICE
    for bt in range(len(btl)):
        for st in range(len(stl)):
            perf = perf[bt, st, BT_IX].assign(btl[bt])
            perf = perf[bt, st, ST_IX].assign(stl[st])

    return (perf, conf)


def best_np(perf):
    (PERF, COUNT, BT_IX, ST_IX) = (ix for ix in range(4))
    bp = -999999.9
    bc = bbt = bst = 0
    btlen, stlen, _ = perf.shape
    for bt in range(btlen):
        for st in range(stlen):
            if perf[bt, st, PERF] > bp:
                (bp, bc, bbt, bst) = \
                    (float(perf[bt, st, PERF]), int(perf[bt, st, COUNT]),
                     float(perf[bt, st, BT_IX]), float(perf[bt, st, ST_IX]))
    return (bp, bbt, bst, int(bc))


def _prep_perf_mat(perf_np):
    (PERF, COUNT, BT_IX, ST_IX) = (ix for ix in range(4))
    btlen, stlen, _ = perf_np.shape
    btl = [float(perf_np[bt, 0, BT_IX]) for bt in range(btlen)]
    stl = [float(perf_np[0, st, ST_IX]) for st in range(stlen)]
    perf_mat = pd.DataFrame(index=btl, columns=pd.MultiIndex.from_product([stl, ["perf", "count"]]))
    perf_mat.index.rename("bt", inplace=True)
    perf_mat.columns.rename(["st", "kpi"], inplace=True)
    for bt in range(btlen):
        for st in range(stlen):
            perf_mat.loc[btl[bt], (stl[st], "perf")] = "{:>5.0%}".format(float(perf_np[bt, st, PERF]))
            perf_mat.loc[btl[bt], (stl[st], "count")] = int(perf_np[bt, st, COUNT])
    return perf_mat


def report_assessment_np(perf, conf, name):
    (PERF, COUNT, BT, ST) = (ix for ix in range(4))
    conf_mat = pd.DataFrame(index=ct.TARGETS.keys(), columns=ct.TARGETS.keys(), dtype=int)
    conf_mat.index.rename("target", inplace=True)
    conf_mat.columns.rename("actual", inplace=True)
    for target_label in ct.TARGETS:
        for actual_label in ct.TARGETS:
            conf_mat.loc[target_label, actual_label] = \
                int(conf[ct.TARGETS[target_label], ct.TARGETS[actual_label]])
    # logger.info(f"confusion matrix np: \n{conf}")
    logger.info(f"{name} confusion matrix overall: \n{conf_mat}")
    (bp, bbt, bst, bc) = best_np(perf)
    logger. info(f"best {name} performance overall {bp:>5.0%} at bt/st {bbt}/{bst} with {bc} transactions")
    # logger.info(f"performance matrix np: \n{perf}")
    logger.info(f"{name} performance matrix overall: \n{_prep_perf_mat(perf)}")


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
    perf, conf = assess_prediction_np(pred_df.values, df["close"].values, df["target"].values)

    report_assessment_np(perf, conf, "test")
    print(perf, conf)

    # W = tf.Variable(tf.ones(shape=(2, 2)), name="W")
    # b = tf.Variable(tf.zeros(shape=(2)), name="b")

    # out_a = forward([1, 0])
    # print(out_a)
