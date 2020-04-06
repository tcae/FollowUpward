#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylama:{name1}={value1}:{name2}={value2}:...
import logging
import pandas as pd
from datetime import timedelta
# from datetime import datetime
import timeit

# import logging
import time
import env_config as env
# from env_config import Env
import crypto_targets as ct
# import crypto_features as cf
# import classify_keras as ck
# from classify_keras import PerfMatrix, EvalPerf  # required for pickle  # noqa
# from local_xch import Xch
from bookkeeping import Bk
import crypto_history_sets as chs
import cached_crypto_data as ccd
import condensed_features as cof
import aggregated_features as agf
import prediction_data as pdd

logger = logging.getLogger(__name__)

MAX_MINBELOW = 5  # max minutes below buy price before emergency sell
ORDER_TIMEOUT = 45  # in seconds
BLOCKED_SELL_ASSET_AMOUNT = {"BNB": 15}  # this amount won't be sold
MPH = 60  # Minutes Per Hour
ICEBERG_USDT_PART = 450
MAX_USDT_ORDER = ICEBERG_USDT_PART * 10  # binanace limitP10 iceberg parts per order


class Trading():
    """To simplify trading some constraints and a workflow are defined.

    Constraints:
    ============
    considered currencies are
    - part of a constant defined set BASES or
    - they are part of the bookkeeping with > 1 USDT or
    - the currency is tradable in USDT and has a daily volume >= MIN_AVG_USDT*60*24
    - shall be tradable with USDT as quote currency --> no need to worry about quote
    - currently binance is the only supported exchange --> no need to worry about the exchange

    Workflow:
    =========
    - considered currencies have an entry in book DataFrame
    - each minute OHLCV is called to provide a DataFrame that is used to classifiy signals
    - a percentage of the average minute volume is the upper bound to buy
    - for sell/buy orders all considered tickers are updated as the last bid/ask is needed anyhow
    - check order progress is at least called once per minute in case of open orders
    - summary: OHLCV > classify > sell/buy > check order progress

    TODO: async execution4
    """

    def __init__(self):
        # FIXME: store order with attributes in order pandas
        self.openorders = list()
        Bk()

    def check_order_progress(self):
        """Check fill status of open orders and adapt price if required.

        current order strategy:
        =======================
        - place iceberg order for sell with best bid (independent on volume)
        - place iceberg order for buy with best ask (independent on volume)
        - cancel open orders after a minute and issue new orders according to signal

        TODO: Order strategy:
        ===============
        - >60% maker trades in last fetch_trades (1-3 minutes):
          30% iceberg order on edge of maker side
          and 60% whale order well on maker side but in front of volumes > 0.5 own volume
        - <= 60% maker trades volume in last fetch_trades (1-3 minutes)
          and price decline < abs(0,1%/minute):
          60% iceberg on maker edge and 40% edge taker.
        - <= 60% maker trades in last fetch_trades (1-3 minutes)
          and price decline > abs(0,1%/minute):
          5% taker every 5 seconds.

        TODO: ML strategy:
        ============
        - collect fetch_trades() and fetch_order_book() as preprocessing input
        - divide book in 1/10 quantile for ask and bid that serve with their voume as 20 features
        - They are the to be predicted classes, i.e. volume in % per quantile in the next minute
        - Fetched trades are mapped to the book quantiles and also serve as 20 features
        - targets are derived by mapping the actual fetched trades to the predicted quantiles
        - only samples are used for training that correlate with a buy or sell signal of that market

        """
        for base in Bk.book.index:
            if Bk.book.loc[base, "used"] > 0:
                oo = Bk.myfetch_open_orders(base, "check_order_progress")
                if oo is not None:
                    for order in oo:
                        is_sell = (order["side"] == "sell")
                        ots = pd.Timestamp(order["datetime"])
                        nowts = pd.Timestamp.utcnow()
                        tsdiff = int((nowts - ots) / timedelta(seconds=1))
                        logger.info(f"now {env.timestr(nowts)} - ts {env.timestr(ots)} = {tsdiff}s")
                        """
                        # ! reconsider why several cancel orders commands for the same order
                        if tsdiff >= ORDER_TIMEOUT:
                            try:
                                Bk.myfetch_cancel_orders(order["id"], base)
                                Bk.myfetch_cancel_orders(order["id"], base)
                            except ccxt.NetworkError:
                                Bk.myfetch_cancel_orders(order["id"], base)
                            except ccxt.OrderNotFound:
                                # that's what we are looking for
                                pass
                        """
                        if tsdiff >= ORDER_TIMEOUT:
                            Bk.myfetch_cancel_orders(order["id"], base)
                            if is_sell:
                                self.sell_order(base, ratio=1)

    def _deduct_blocked_sell_amount(self, base, base_amount):
        base = base.upper()
        if base in BLOCKED_SELL_ASSET_AMOUNT:
            base_amount -= BLOCKED_SELL_ASSET_AMOUNT[base]
        return base_amount

    def sell_order(self, base, ratio=1):
        """ Sells the ratio of free base currency. Constraint: 0 <= ratio <= 1
        """
        assert 0 <= ratio <= 1, f"not compliant to constraint: 0 <= ratio {ratio} <= 1"
        isincheck = True in Bk.book.index.isin([base])
        if isincheck:
            sym = Bk.xhc_sym_of_base(base)
            tickers = Bk.myfetch_tickers("sell_order")
            if tickers is None:
                return
            Bk.update_balance(tickers)
            base_amount = Bk.book.loc[base, "free"]
            base_amount = self._deduct_blocked_sell_amount(base, base_amount)
            base_amount *= ratio
            price = tickers[sym]["bid"]  # TODO order spread strategy
            logger.info(f"SELL {base_amount} {base} x {price} {sym}")
            while base_amount > 0:
                price = tickers[sym]["bid"]  # TODO order spread strategy
                max_chunk = MAX_USDT_ORDER / price
                ice_chunk = amount = min([base_amount, max_chunk])
                (amount, price, ice_chunk) = Bk.check_limits(base, amount, price)
                if (amount == 0) or (price == 0):
                    return
                myorder = Bk.create_limit_sell_order(
                    sym, amount, price,
                    {"icebergQty": ice_chunk, "timeInForce": "GTC"})
                if myorder is None:
                    return
                logger.info(myorder)
                # FIXME: store order with attributes in order pandas
                self.openorders.append(myorder)
                base_amount -= amount
        else:
            logger.info(f"unsupported base {base}")

    def buy_order(self, base, ratio=1):
        """ Buys the ratio of free quote currency with base currency. Constraint: 0 <= ratio <= 1
        """
        assert 0 <= ratio <= 1, f"not compliant to constraint: 0 <= ratio {ratio} <= 1"
        isincheck = True in Bk.book.index.isin([base])
        if isincheck:
            sym = Bk.xhc_sym_of_base(base)
            tickers = Bk.myfetch_tickers("buy_order")
            if tickers is None:
                return

            Bk.update_balance(tickers)
            quote_amount = Bk.trade_amount(base, ratio)
            price = tickers[sym]["ask"]  # TODO order spread strategy
            logger.info(f"BUY {ratio} ratio = {quote_amount} USDT / {price} {sym}")
            while quote_amount > 0:
                price = tickers[sym]["ask"]  # TODO order spread strategy
                max_chunk = MAX_USDT_ORDER / price
                ice_chunk = amount = min([quote_amount/price, max_chunk])
                (amount, price, ice_chunk) = Bk.check_limits(base, amount, price)
                if (amount == 0) or (price == 0):
                    return
                myorder = Bk.create_limit_buy_order(
                    base, amount, price,
                    {"icebergQty": ice_chunk, "timeInForce": "GTC"})
                if myorder is None:
                    return
                logger.info(myorder)
                # FIXME: store order with attributes in order pandas
                self.openorders.append(myorder)
                quote_amount -= amount * price
        else:
            logger.info(f"unsupported base {base}")

    def _force_sell_check(self, base, last_close, trade_signal):
        """ Changes trade_signal into a forced sell if the classifier fails
            and the close price is below the buy price for more
            than a configurable number of minutes MAX_MINBELOW
        """
        if trade_signal != ct.TARGETS[ct.BUY]:
            # emergency sell in case no SELL signal but performance drops
            if Bk.book.loc[base, "buyprice"] > 0:
                if Bk.book.loc[base, "buyprice"] > last_close:
                    Bk.book.loc[base, "minbelow"] += 1  # minutes below buy price
                else:
                    Bk.book.loc[base, "minbelow"] = 0  # reset minute counter
                if Bk.book.loc[base, "minbelow"] > MAX_MINBELOW:  # minutes
                    sym = Bk.xhc_sym_of_base(base)
                    logger.info("!!! Forced selling {} due to {} minutes < buy price of {}".format(
                            sym,
                            Bk.book.loc[base, "minbelow"],
                            Bk.book.loc[base, "buyprice"]))
                    trade_signal = ct.TARGETS[ct.SELL]
                    Bk.book.loc[base, "buyprice"] = 0  # reset price monitoring
                    Bk.book.loc[base, "minbelow"] = 0  # minutes below buy price
        return trade_signal

    def _get_signal(self, cpc, buy_trshld, sell_trshld, base, date_time):
        """ Encapsulates all handling to get the trade signal of a base at a
            given python datetime
        """
        if base in Bk.black_bases:  # USDT is in Bk.book
            return None, 0
        ohlcv_df = Bk.get_ohlcv(base, chs.ActiveFeatures.history(), date_time)
        if ohlcv_df is None:
            logger.info(f"skipping {base} due to missing ohlcv")
            return None, 0
        ttf = chs.ActiveFeatures(base, minute_dataframe=ohlcv_df)
        ttf.vec = ttf.calc_features(ttf.minute_data)
        # ttf.calc_features_and_targets()  --> don't calc targets
        tfv = ttf.vec.iloc[[len(ttf.vec)-1]]
        trade_signal = cpc.class_of_features(tfv, buy_trshld, sell_trshld, base)
        # trade_signal will be HOLD if insufficient data history is available
        last_close = ohlcv_df.loc[ohlcv_df.index[len(ohlcv_df.index)-1], "close"]
        return trade_signal, last_close

    def _distribute_buy_amount(self, buylist):
        """ distributes the available buy amount among collected buy orders candidates
        """
        if len(buylist) > 0:
            free_distribution = 1/len(buylist)  # equally distributed weight
            # TODO: 50% of weight via Alpha of that currency
            for base in buylist:
                self.buy_order(base, ratio=free_distribution)
            buylist.clear()

    def trade_loop(self, cpc, buy_trshld, sell_trshld):
        """ endless loop than can be interrupted by Ctrl-C that
            executes the trading logic
        """
        buylist = list()
        try:
            while True:
                logger.info(f"next round")
                # TOD: check order progress
                ts1 = pd.Timestamp.utcnow()
                for base in Bk.book.index:
                    trade_signal, last_close = self._get_signal(
                        cpc, buy_trshld, sell_trshld, base, ts1)
                    if trade_signal is None:
                        continue

                    trade_signal = self._force_sell_check(base, last_close, trade_signal)
                    if trade_signal != ct.TARGETS[ct.HOLD]:
                        logger.info(f"{base} {ct.TARGET_NAMES[trade_signal]}")
                    if trade_signal == ct.TARGETS[ct.SELL]:
                        self.sell_order(base, ratio=1)
                    if trade_signal == ct.TARGETS[ct.BUY]:
                        buylist.append(base)  # amount to be determined by __distribute_buy_amount

                self._distribute_buy_amount(buylist)
                ts2 = pd.Timestamp.utcnow()
                tsdiff = 59 - int((ts2 - ts1) / pd.Timedelta(1, unit='S'))  # 1 seconds order progress
                if tsdiff > 1:
                    time.sleep(tsdiff)
                self.check_order_progress()
        except KeyboardInterrupt:
            Bk.safe_cache()
            logger.info(Bk.actions)
            logger.info("finish as requested by keyboard interrupt")


def trading_main():
    # env.set_environment(env.Usage.test, env.Calc.ubuntu)
    tee = env.Tee(log_prefix="Trading")
    trading = Trading()
    ohlcv = ccd.Ohlcv()
    targets = ct.T10up5low30min(ohlcv)
    if True:
        features = cof.F3cond14(ohlcv)
    else:
        features = agf.AggregatedFeatures(ohlcv)
    classifier = pdd.Classifier(ohlcv, features, targets)
    classifier.load("MLP2_talos_iter-3_l1-14_do-0.2_l2-16_l3-no_opt-Adam__F2cond20__T10up5low30min", epoch=15)

    start_time = timeit.default_timer()
    buy_trshld = 0.7
    sell_trshld = 0.8

    # trd.buy_order("ETH", ratio=22/trd.book.loc[Env.quote, "free"])
    # trd.sell_order("ETH", ratio=1)
    trading.trade_loop(classifier, buy_trshld, sell_trshld)
    trading = None  # should trigger Trading destructor

    tdiff = (timeit.default_timer() - start_time) / (60*60)
    logger.info(f"total time: {tdiff:.2f} hours")
    tee.close()


if __name__ == "__main__":
    trading_main()
