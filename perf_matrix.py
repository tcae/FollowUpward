import timeit
import numpy as np
import pandas as pd
import logging
# import cached_crypto_data as ccd
import crypto_targets as ct

logger = logging.getLogger(__name__)


class EvalPerf:
    """Evaluates the performance of a buy/sell probability threshold
    """

    def __init__(self, buy_prob_threshold, sell_prob_threshold):
        """receives minimum thresholds to consider the transaction
        """
        self.bpt = buy_prob_threshold
        self.spt = sell_prob_threshold
        self.transactions = 0
        self.open_transaction = False
        self.performance = 0

    def add_trade_signal(self, prob, close_price, signal):
        if signal == ct.TARGETS[ct.BUY]:
            if not self.open_transaction:
                if prob >= self.bpt:
                    self.open_transaction = True
                    self.open_buy = close_price * (1 + ct.FEE)
                    self.highest = close_price
        elif signal == ct.TARGETS[ct.SELL]:
            if self.open_transaction:
                if prob >= self.spt:
                    self.open_transaction = False
                    gain = (close_price * (1 - ct.FEE) - self.open_buy) / self.open_buy
                    self.performance += gain
                    self.transactions += 1
        elif signal == ct.TARGETS[ct.HOLD]:
            pass

    def __str__(self):
        return "bpt: {:<5.2} spt: {:<5.2} perf: {:>5.0%} #trans.: {:>4}".format(
              self.bpt, self.spt, self.performance, self.transactions)


class PerfMatrix:
    """Evaluates the performance across a range of buy/sell thresholds
    """

    def __init__(self, epoch, set_type="unknown set type"):

        self.p_range = range(6, 10)
        self.perf = np.zeros((len(self.p_range), len(self.p_range)), dtype=EvalPerf)
        for bp in self.p_range:
            for sp in self.p_range:
                self.perf[bp - self.p_range[0], sp - self.p_range[0]] = \
                    EvalPerf(float((bp)/10), float((sp)/10))
        self.confusion = np.zeros((ct.TARGET_CLASS_COUNT, ct.TARGET_CLASS_COUNT), dtype=int)
        self.epoch = epoch
        self.set_type = set_type
        self.start_ts = timeit.default_timer()
        self.end_ts = None

        self.BUY = ct.TARGETS[ct.BUY]
        self.SELL = ct.TARGETS[ct.SELL]
        self.HOLD = ct.TARGETS[ct.HOLD]
        self.track = [self.PERF, self.COUNT, self.BUY_IX, self.BUY_PRICE] = [ix for ix in range(4)]
        self.btl = [0.6, 0.7, 0.8, 0.9]
        self.stl = [0.6, 0.7, 0.8, 0.9]
        self.perf2 = np.zeros((len(self.btl), len(self.stl), len(self.track)))
        self.conf2 = np.zeros((len(ct.TARGETS), len(ct.TARGETS)))  # (target, actual)

    def pix(self, bp, sp):
        return self.perf[bp - self.p_range[0], sp - self.p_range[0]]

    def print_result_distribution(self):
        for bp in self.p_range:
            for sp in self.p_range:
                print(self.pix(bp, sp))

    """
    def plot_result_distribution(self):
        dtype = [("bpt", float), ("spt", float), ("performance", float), ("transactions", int)]
        values = list()
        for bp in self.p_range:
            for sp in self.p_range:
                values.append((float((bp)/10), float((sp)/10),
                               self.pix(bp, sp).performance, self.pix(bp, sp).transactions))
                perf_result = np.array(values, dtype)
                perf_result = np.sort(perf_result, order="performance")
        plt.figure()
        plt.title("transactions over performance")
        maxt = np.amax(perf_result["transactions"])
        maxp = np.amax(perf_result["performance"])
        plt.ylim((0., maxp))
        plt.ylabel("performance")
        plt.xlabel("transactions")
        plt.grid()
        xaxis = np.arange(0, maxt, maxt*0.1)
        plt.plot(xaxis, perf_result["performance"], "o-", color="g",
                 label="performance")
        plt.legend(loc="best")
        plt.show()
    """

    def best(self):
        """Returns a tuple of (best-performance-factor, at-buy-probability-threshold,
        at-sell-probability-threshold, with-number-of-transactions)
        """
        best = float(self.perf[0, 0].performance)
        hbp = hsp = self.p_range[0]
        for bp in self.p_range:
            for sp in self.p_range:
                if best < self.pix(bp, sp).performance:
                    best = self.pix(bp, sp).performance
                    hbp = bp
                    hsp = sp
        bpt = float(hbp/10)
        spt = float(hsp/10)
        t = self.pix(hbp, hsp).transactions
        return (best, bpt, spt, t)

    def __str__(self):
        (best, bpt, spt, t) = self.best()
        return f"epoch {self.epoch}, best performance {best:6.0%}" + \
            f" with buy threshold {bpt:.1f} / sell threshold {spt:.1f} at {t} transactions"

    def add_signal(self, prob, close_price, signal, target):
        assert (prob >= 0) and (prob <= 1), \
                logger.warning(f"PerfMatrix add_signal: unexpected probability {prob}")
        if signal in ct.TARGETS.values():
            for bp in self.p_range:
                for sp in self.p_range:
                    self.pix(bp, sp).add_trade_signal(prob, close_price, signal)
            if target not in ct.TARGETS.values():
                logger.error(f"PerfMatrix add_signal: unexpected target result {target}")
                return
            self.confusion[signal, target] += 1
        else:
            raise ValueError(f"PerfMatrix add_signal: unexpected class result {signal}")

    def assess_sample_prediction(self, pred, skl_close, skl_target, skl_tics):
        """Assess the highest probability of a class prediction
        to find the optimum buy/sell threshold. No consideration of time gaps or open transactions
        at the end of the sample sequence
        """
        pred_cnt = len(pred)
        # begin = env.timestr(skl_tics[0])
        for sample in range(pred_cnt):
            high_prob_cl = 0
            for cl in range(len(pred[0])):
                if pred[sample, high_prob_cl] < pred[sample, cl]:
                    high_prob_cl = cl
            self.add_signal(pred[sample, high_prob_cl], skl_close[sample],
                            high_prob_cl, skl_target[sample])

    def assess_prediction(self, pred_np, close_df, target_df):
        """Assess the highest probability of a class prediction
        to find the optimum buy/sell threshold.
        pred_df has a datetime index and columns close, buy and sell.

        - close = close price
        - buy = buy likelihood
        - sell = sell likelihood

        handling of time gaps: in case of a time gap the last value of the time slice is taken
        to close any open transaction

        """
        pred_cnt = len(pred_np)
        tic_arr = close_df.index.values
        close_arr = close_df["close"].values
        target_arr = target_df["target"].values
        # begin = env.timestr(skl_tics[0])
        for sample in range(pred_cnt):
            if (sample + 1) >= pred_cnt:
                self.add_signal(1, close_arr[sample], ct.TARGETS[ct.SELL], ct.TARGETS[ct.SELL])
                # end = env.timestr(skl_tics[sample])
                # logger.debug("assessment between {} and {}".format(begin, end))
            elif (tic_arr[sample+1] - tic_arr[sample]) > np.timedelta64(1, "m"):
                self.add_signal(1, close_arr[sample], ct.TARGETS[ct.SELL], ct.TARGETS[ct.SELL])
                # end = env.timestr(skl_tics[sample])
                # logger.debug("assessment between {} and {}".format(begin, end))
                # begin = env.timestr(skl_tics[sample+1])
            else:
                if pred_np[sample, ct.TARGETS[ct.HOLD]] > pred_np[sample, ct.TARGETS[ct.BUY]]:
                    if pred_np[sample, ct.TARGETS[ct.HOLD]] > pred_np[sample, ct.TARGETS[ct.SELL]]:
                        self.add_signal(pred_np[sample, ct.TARGETS[ct.HOLD]], close_arr[sample],
                                        ct.TARGETS[ct.HOLD], target_arr[sample])
                    else:
                        self.add_signal(pred_np[sample, ct.TARGETS[ct.SELL]], close_arr[sample],
                                        ct.TARGETS[ct.SELL], target_arr[sample])
                else:
                    if pred_np[sample, ct.TARGETS[ct.BUY]] > pred_np[sample, ct.TARGETS[ct.SELL]]:
                        self.add_signal(pred_np[sample, ct.TARGETS[ct.BUY]], close_arr[sample],
                                        ct.TARGETS[ct.BUY], target_arr[sample])
                    else:
                        self.add_signal(pred_np[sample, ct.TARGETS[ct.SELL]], close_arr[sample],
                                        ct.TARGETS[ct.SELL], target_arr[sample])

    def assess_prediction_np(self, pred_np, close_df, target_df):  # noqa
        """Assess the highest probability of a class prediction
        to find the optimum buy/sell threshold.
        pred_df has a datetime index and columns close, buy and sell.

        - close = close price
        - buy = buy likelihood
        - sell = sell likelihood

        handling of time gaps: in case of a time gap the last value of the time slice is taken
        to close any open transaction

        """
        pred_cnt = len(pred_np)
        # tic_arr = close_df.index.values
        close_arr = close_df["close"].values
        target_arr = target_df["target"].values
        btl_len = len(self.btl)
        stl_len = len(self.stl)
        BUY = self.BUY
        SELL = self.SELL
        HOLD = self.HOLD
        # begin = env.timestr(skl_tics[0])
        for sample in range(pred_cnt):
            pred_sell = pred_np[sample, SELL]
            pred_buy = pred_np[sample, BUY]
            pred_hold = pred_np[sample, HOLD]
            if pred_sell >= pred_buy:
                if pred_sell > pred_hold:
                    # SELL
                    for bt in range(btl_len):
                        for st in range(stl_len):
                            buy_price = self.perf2[bt, st, self.BUY_PRICE]
                            if pred_sell > self.stl[st]:
                                if buy_price > 0:
                                    # add here the transaction tracking shall be inserted
                                    transaction_perf = \
                                        (close_arr[sample] * (1 - ct.FEE) -
                                         buy_price * (1 + ct.FEE)) \
                                        / buy_price
                                    self.perf2[bt, st, self.PERF] += transaction_perf
                                    self.perf2[bt, st, self.COUNT] += 1
                                    self.perf2[bt, st, self.BUY_PRICE] = 0
                    self.conf2[target_arr[sample], SELL] += 1
                else:
                    # HOLD
                    self.conf2[target_arr[sample], HOLD] += 1
            else:  # SELL < BUY
                if pred_buy > pred_hold:
                    # BUY
                    for bt in range(btl_len):
                        for st in range(stl_len):
                            if pred_buy > self.btl[bt]:
                                self.perf2[bt, st, self.BUY_PRICE] = close_arr[sample]
                                # self.perf2[bt, st, self.BUY_IX] = sample
                    self.conf2[target_arr[sample], BUY] += 1
                else:
                    # HOLD
                    self.conf2[target_arr[sample], HOLD] += 1

    def close_open_transactions_np(self, close_df):
        close_arr = close_df["close"].values
        for bt in range(len(self.btl)):
            for st in range(len(self.stl)):
                buy_price = self.perf2[bt, st, self.BUY_PRICE]
                if buy_price > 0:
                    # add here the transaction tracking
                    transaction_perf = \
                        (close_arr[-1] * (1 - ct.FEE) -
                            buy_price * (1 + ct.FEE)) \
                        / buy_price
                    self.perf2[bt, st, self.PERF] += transaction_perf
                    self.perf2[bt, st, self.COUNT] += 1
                    self.perf2[bt, st, self.BUY_PRICE] = 0

    def best_np(self):
        bp = -999999.9
        bc = bbt = bst = 0
        for bt in range(len(self.btl)):
            for st in range(len(self.stl)):
                if self.perf2[bt, st, self.PERF] > bp:
                    (bp, bc, bbt, bst) = (self.perf2[bt, st, self.PERF],
                                          self.perf2[bt, st, self.COUNT], self.btl[bt], self.stl[st])
        return (bp, bbt, bst, bc)

    def report_assessment_np(self):
        perf_mat = pd.DataFrame(index=self.btl, columns=pd.MultiIndex.from_product([self.stl, ["perf", "count"]]))
        perf_mat.index.rename("bt", inplace=True)
        perf_mat.columns.rename(["st", "kpi"], inplace=True)
        conf_mat = pd.DataFrame(index=ct.TARGETS.keys(), columns=ct.TARGETS.keys(), dtype=int)
        conf_mat.index.rename("target", inplace=True)
        conf_mat.columns.rename("actual", inplace=True)
        for bt in range(len(self.btl)):
            for st in range(len(self.stl)):
                perf_mat.loc[self.btl[bt], (self.stl[st], "perf")] = "{:>5.0%}".format(self.perf2[bt, st, self.PERF])
                perf_mat.loc[self.btl[bt], (self.stl[st], "count")] = int(self.perf2[bt, st, self.COUNT])
        for target_label in ct.TARGETS:
            for actual_label in ct.TARGETS:
                conf_mat.loc[target_label, actual_label] = \
                    int(self.conf2[ct.TARGETS[target_label], ct.TARGETS[actual_label]])
        (bp, bbt, bst, bc) = self.best_np()
        logger. info(f"best performance {bp} at bt/st {bbt}/{bst} with {bc} transactions")
        # logger.info(f"performance matrix np: \n{self.perf2}")
        logger.info(f"performance matrix np df: \n{perf_mat}")
        # logger.info(f"confusion matrix np: \n{self.conf2}")
        logger.info(f"confusion matrix np df: \n{conf_mat}")

    def conf(self, estimate_class, target_class):
        elem = self.confusion[estimate_class, target_class]
        targets = 0
        estimates = 0
        for i in ct.TARGETS:
            targets += self.confusion[estimate_class, ct.TARGETS[i]]
            estimates += self.confusion[ct.TARGETS[i], target_class]
        return (elem, elem/estimates, elem/targets)

    def report_assessment(self):
        self.end_ts = timeit.default_timer()
        tdiff = (self.end_ts - self.start_ts) / 60
        print(f"{self.set_type} performance assessment time: {tdiff:.1f}min")

        def pt(bp, sp): return (self.pix(bp, sp).performance, self.pix(bp, sp).transactions)

        print(self)
        print("target:    {: >7}/est%/tgt% {: >7}/est%/tgt% {: >7}/est%/tgt%".format(
                ct.HOLD, ct.BUY, ct.SELL))
        print("est. {: >4}: {: >7}/{:4.0%}/{:4.0%} {: >7}/{:4.0%}/{:4.0%} {: >7}/{:4.0%}/{:4.0%}".format(
              ct.HOLD,
              *self.conf(ct.TARGETS[ct.HOLD], ct.TARGETS[ct.HOLD]),
              *self.conf(ct.TARGETS[ct.HOLD], ct.TARGETS[ct.BUY]),
              *self.conf(ct.TARGETS[ct.HOLD], ct.TARGETS[ct.SELL])))
        print("est. {: >4}: {: >7}/{:4.0%}/{:4.0%} {: >7}/{:4.0%}/{:4.0%} {: >7}/{:4.0%}/{:4.0%}".format(
              ct.BUY,
              *self.conf(ct.TARGETS[ct.BUY], ct.TARGETS[ct.HOLD]),
              *self.conf(ct.TARGETS[ct.BUY], ct.TARGETS[ct.BUY]),
              *self.conf(ct.TARGETS[ct.BUY], ct.TARGETS[ct.SELL])))
        print("est. {: >4}: {: >7}/{:4.0%}/{:4.0%} {: >7}/{:4.0%}/{:4.0%} {: >7}/{:4.0%}/{:4.0%}".format(
              ct.SELL,
              *self.conf(ct.TARGETS[ct.SELL], ct.TARGETS[ct.HOLD]),
              *self.conf(ct.TARGETS[ct.SELL], ct.TARGETS[ct.BUY]),
              *self.conf(ct.TARGETS[ct.SELL], ct.TARGETS[ct.SELL])))

        # self.print_result_distribution()
        print("performance matrix: estimated probability/number of buy+sell trades")
        print("     {: ^10} {: ^10} {: ^10} {: ^10} < spt".format(0.6, 0.7, 0.8, 0.9))
        print("0.6  {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4}".format(
                *pt(6, 6), *pt(6, 7), *pt(6, 8), *pt(6, 9)))
        print("0.7  {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4}".format(
                *pt(7, 6), *pt(7, 7), *pt(7, 8), *pt(7, 9)))
        print("0.8  {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4}".format(
                *pt(8, 6), *pt(8, 7), *pt(8, 8), *pt(8, 9)))
        print("0.9  {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4}".format(
                *pt(9, 6), *pt(9, 7), *pt(9, 8), *pt(9, 9)))
        print("^bpt")


if __name__ == "__main__":
    track = [PERF, COUNT, BUY_IX, BUY_PRICE] = [ix for ix in range(4)]
    print(track)
    btl = [0.61, 0.7]
    stl = [0.5, 0.6]
    x = [(bt, st) for bt in btl for st in stl]
    print(f"bt/st {x}")
