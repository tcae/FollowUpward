# import timeit
import numpy as np
import pandas as pd
import logging
# import cached_crypto_data as ccd
import crypto_targets as ct

logger = logging.getLogger(__name__)


class PerfMatrix:
    """Evaluates the performance across a range of buy/sell thresholds
    """

    def __init__(self, epoch, set_type="unknown set type"):

        self.epoch = epoch
        self.set_type = set_type
        self.BUY = ct.TARGETS[ct.BUY]
        self.SELL = ct.TARGETS[ct.SELL]
        self.HOLD = ct.TARGETS[ct.HOLD]
        self.track = [self.PERF, self.COUNT, self.BUY_IX, self.BUY_PRICE] = [ix for ix in range(4)]
        self.btl = [0.6, 0.7, 0.8, 0.9]  # buy thresholds
        self.stl = [0.6, 0.7, 0.8, 0.9]  # sell thresholds
        self.perf = np.zeros((len(self.btl), len(self.stl), len(self.track)))  # overall performace array
        self.base_perf = dict()  # performance array for each base
        self.conf = np.zeros((len(ct.TARGETS), len(ct.TARGETS)))  # (target, actual) confusion matrix

    def _process_sell(self, base, close_arr, sample, pred_sell):
        """ process sell signal without confusion matrix update
        """
        for st in range(len(self.stl)):
            if pred_sell >= self.stl[st]:
                for bt in range(len(self.btl)):
                    buy_price = self.perf[bt, st, self.BUY_PRICE]
                    if buy_price > 0:
                        # add here the transaction tracking shall be inserted
                        transaction_perf = \
                            (close_arr[sample] * (1 - ct.FEE) - buy_price * (1 + ct.FEE)) \
                            / buy_price * (1 + ct.FEE)
                        self.perf[bt, st, self.PERF] += transaction_perf
                        self.perf[bt, st, self.COUNT] += 1
                        self.perf[bt, st, self.BUY_PRICE] = 0
                        self.base_perf[base][bt, st, self.PERF] += transaction_perf
                        self.base_perf[base][bt, st, self.COUNT] += 1

    def _process_buy(self, base, close_arr, sample, pred_buy):
        """ process sell signal without confusion matrix update
        """
        for bt in range(len(self.btl)):
            if pred_buy >= self.btl[bt]:
                for st in range(len(self.stl)):
                    if self.perf[bt, st, self.BUY_PRICE] == 0:
                        # first buy of a possible sequence of multiple buys before sell
                        self.perf[bt, st, self.BUY_PRICE] = close_arr[sample]
                        self.perf[bt, st, self.BUY_IX] = sample

    def _close_open_transactions_np(self, base, close_df, target_df):
        """ only corrects the performance calculation but not the confusion matrix
        """
        close_arr = close_df["close"].values
        for bt in range(len(self.btl)):
            for st in range(len(self.stl)):
                buy_price = self.perf[bt, st, self.BUY_PRICE]
                if buy_price > 0:
                    if self.perf[bt, st, self.BUY_IX] == (len(target_df) - 1):
                        # last transaction was opened by last sample --> revert transaction
                        self.perf[bt, st, self.BUY_PRICE] = 0
                    else:
                        # last transaction was opened before last sample --> close with sell
                        # add here the transaction tracking
                        transaction_perf = \
                            (close_arr[-1] * (1 - ct.FEE) - buy_price * (1 + ct.FEE)) / buy_price * (1 + ct.FEE)
                        self.perf[bt, st, self.PERF] += transaction_perf
                        self.perf[bt, st, self.COUNT] += 1
                        self.perf[bt, st, self.BUY_PRICE] = 0
                        self.base_perf[base][bt, st, self.PERF] += transaction_perf
                        self.base_perf[base][bt, st, self.COUNT] += 1

    def assess_prediction_np(self, base: str, pred_np, close_df, target_df):
        """ Assesses the performance with different buy/sell thresholds. This method can be called multiple times,
            which accumulates the performance results in the class attributes 'perf' and 'conf'

            - pred_np is a tensor with numpy data returning the predictions per class
              with class index == ct.TARGETS[BUY|SELL|HOLD]
            - close_df is a data frame with a 'close' column containing close prices
            - target_df is a data frame with a 'target' column containing the target class index
        """
        pred_cnt = len(pred_np)
        close_arr = close_df["close"].values
        target_arr = target_df["target"].values
        BUY = self.BUY
        SELL = self.SELL
        HOLD = self.HOLD
        if base not in self.base_perf:
            self.base_perf[base] = np.zeros((len(self.btl), len(self.stl), len(self.track)))
        for sample in range(pred_cnt):
            pred_sell = pred_np[sample, SELL]
            pred_buy = pred_np[sample, BUY]
            pred_hold = pred_np[sample, HOLD]
            if pred_sell >= pred_buy:
                if pred_sell >= pred_hold:
                    # SELL
                    self._process_sell(base, close_arr, sample, pred_sell)
                    self.conf[target_arr[sample], SELL] += 1
                else:
                    # HOLD
                    self.conf[target_arr[sample], HOLD] += 1
            else:  # SELL < HOLD
                if pred_buy >= pred_hold:
                    # BUY
                    self._process_buy(base, close_arr, sample, pred_buy)
                    self.conf[target_arr[sample], BUY] += 1
                else:
                    # HOLD
                    self.conf[target_arr[sample], HOLD] += 1
        self._close_open_transactions_np(base, close_df, target_df)

    def best_np(self):
        bp = -999999.9
        bc = bbt = bst = 0
        for bt in range(len(self.btl)):
            for st in range(len(self.stl)):
                if self.perf[bt, st, self.PERF] > bp:
                    (bp, bc, bbt, bst) = (self.perf[bt, st, self.PERF],
                                          self.perf[bt, st, self.COUNT], self.btl[bt], self.stl[st])
        return (bp, bbt, bst, int(bc))

    def _prep_perf_mat(self, perf_np):
        perf_mat = pd.DataFrame(index=self.btl, columns=pd.MultiIndex.from_product([self.stl, ["perf", "count"]]))
        perf_mat.index.rename("bt", inplace=True)
        perf_mat.columns.rename(["st", "kpi"], inplace=True)
        for bt in range(len(self.btl)):
            for st in range(len(self.stl)):
                perf_mat.loc[self.btl[bt], (self.stl[st], "perf")] = "{:>5.0%}".format(perf_np[bt, st, self.PERF])
                perf_mat.loc[self.btl[bt], (self.stl[st], "count")] = int(perf_np[bt, st, self.COUNT])
        return perf_mat

    def report_assessment_np(self):
        conf_mat = pd.DataFrame(index=ct.TARGETS.keys(), columns=ct.TARGETS.keys(), dtype=int)
        conf_mat.index.rename("target", inplace=True)
        conf_mat.columns.rename("actual", inplace=True)
        for target_label in ct.TARGETS:
            for actual_label in ct.TARGETS:
                conf_mat.loc[target_label, actual_label] = \
                    int(self.conf[ct.TARGETS[target_label], ct.TARGETS[actual_label]])
        # logger.info(f"confusion matrix np: \n{self.conf}")
        logger.info(f"{self.set_type} confusion matrix overall: \n{conf_mat}")
        (bp, bbt, bst, bc) = self.best_np()
        logger. info(f"best {self.set_type} performance overall {bp:>5.0%} at bt/st {bbt}/{bst} with {bc} transactions")
        # logger.info(f"performance matrix np: \n{self.perf}")
        logger.info(f"{self.set_type} performance matrix overall: \n{self._prep_perf_mat(self.perf)}")
        for base in self.base_perf:
            logger.info(f"{self.set_type} performance matrix of {base}: \n{self._prep_perf_mat(self.base_perf[base])}")


if __name__ == "__main__":
    track = [PERF, COUNT, BUY_IX, BUY_PRICE] = [ix for ix in range(4)]
    print(track)
    btl = [0.61, 0.7]
    stl = [0.5, 0.6]
    x = [(bt, st) for bt in btl for st in stl]
    print(f"bt/st {x}")
