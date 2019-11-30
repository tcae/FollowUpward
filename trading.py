#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylama:{name1}={value1}:{name2}={value2}:...
import pandas as pd
from datetime import timedelta
from datetime import datetime
import timeit

# import logging
import time
import env_config as env
import crypto_targets_features as ctf
import classify_keras as ck
from classify_keras import PerfMatrix, EvalPerf  # required for pickle  # noqa
from local_xch import Xch, Bk

MAX_MINBELOW = 0  # max minutes below buy price before emergency sell
ORDER_TIMEOUT = 45  # in seconds
TRADE_VOL_LIMIT_USDT = 100
BLOCKED_ASSET_AMOUNT = {"BNB": 100, "USDT": 20000}  # amounts in base currency
# BASES = ["USDT"]
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
        tickers = Xch.myfetch_tickers("__init__")
        if tickers is not None:
            Bk.update_bookkeeping(tickers)

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
                oo = Xch.myfetch_open_orders(base, "check_order_progress")
                if oo is not None:
                    for order in oo:
                        is_sell = (order["side"] == "sell")
                        ots = pd.Timestamp(order["datetime"])
                        nowts = pd.Timestamp.utcnow()
                        tsdiff = int((nowts - ots) / timedelta(seconds=1))
                        print(f"now {env.timestr(nowts)} - ts {env.timestr(ots)} = {tsdiff}s")
                        """
                        # ! reconsider why several cancel orders commands for the same order
                        if tsdiff >= ORDER_TIMEOUT:
                            try:
                                Xch.myfetch_cancel_orders(order["id"], base)
                                Xch.myfetch_cancel_orders(order["id"], base)
                            except ccxt.NetworkError:
                                Xch.myfetch_cancel_orders(order["id"], base)
                            except ccxt.OrderNotFound:
                                # that's what we are looking for
                                pass
                        """
                        if tsdiff >= ORDER_TIMEOUT:
                            Xch.myfetch_cancel_orders(order["id"], base)
                            if is_sell:
                                self.sell_order(base, ratio=1)

    def sell_order(self, base, ratio=1):
        """ Sells the ratio of free base currency. Constraint: 0 <= ratio <= 1
        """
        assert 0 <= ratio <= 1, f"not compliant to constraint: 0 <= ratio {ratio} <= 1"
        isincheck = True in Bk.book.index.isin([base])
        if isincheck:
            sym = base + "/" + Env.quote
            tickers = Xch.myfetch_tickers("sell_order")
            if tickers is None:
                return
            Bk.update_balance(tickers)
            base_amount = Bk.book.loc[base, "free"]
            if base in BLOCKED_ASSET_AMOUNT:
                base_amount -= BLOCKED_ASSET_AMOUNT[base]
            base_amount *= ratio
            price = tickers[sym]["bid"]  # TODO order spread strategy
            print(f"{env.nowstr()} SELL {base_amount} {base} x {price} {sym}")
            while base_amount > 0:
                price = tickers[sym]["bid"]  # TODO order spread strategy
                max_chunk = MAX_USDT_ORDER / price
                ice_chunk = amount = min([base_amount, max_chunk])
                (amount, price, ice_chunk) = Xch.check_limits(base, amount, price)
                if (amount == 0) or (price == 0):
                    return
                myorder = Xch.create_limit_sell_order(
                    sym, amount, price,
                    {"icebergQty": ice_chunk, "timeInForce": "GTC"})
                if myorder is None:
                    return
                print(myorder)
                # FIXME: store order with attributes in order pandas
                self.openorders.append(myorder)
                base_amount -= amount
        else:
            print(f"unsupported base {base}")

    def trade_amount(self, base, ratio):
        trade_vol = 0
        for base in Bk.book.index:
            if base not in Xch.black_bases:
                trade_vol += (Bk.book.loc[base, "free"] + Bk.book.loc[base, "used"]) * \
                             Bk.book.loc[base, "USDT"]
        trade_vol = TRADE_VOL_LIMIT_USDT - trade_vol
        trade_vol = min(trade_vol, Bk.book.loc[Env.quote, "free"])
        if Env.quote in BLOCKED_ASSET_AMOUNT:
            trade_vol -= BLOCKED_ASSET_AMOUNT[Env.quote]
        usdt_amount = trade_vol * ratio
        return usdt_amount

    def buy_order(self, base, ratio=1):
        """ Buys the ratio of free quote currency with base currency. Constraint: 0 <= ratio <= 1
        """
        assert 0 <= ratio <= 1, f"not compliant to constraint: 0 <= ratio {ratio} <= 1"
        isincheck = True in Bk.book.index.isin([base])
        if isincheck:
            sym = base + "/" + Env.quote
            tickers = Xch.myfetch_tickers("buy_order")
            if tickers is None:
                return

            Bk.update_balance(tickers)
            quote_amount = self.trade_amount(base, ratio)
            price = tickers[sym]["ask"]  # TODO order spread strategy
            print(f"{env.nowstr()} BUY {quote_amount} USDT / {price} {sym}")
            while quote_amount > 0:
                price = tickers[sym]["ask"]  # TODO order spread strategy
                max_chunk = MAX_USDT_ORDER / price
                ice_chunk = amount = min([quote_amount/price, max_chunk])
                (amount, price, ice_chunk) = Xch.check_limits(base, amount, price)
                if (amount == 0) or (price == 0):
                    return
                myorder = Xch.create_limit_buy_order(
                    base, amount, price,
                    {"icebergQty": ice_chunk, "timeInForce": "GTC"})
                if myorder is None:
                    return
                print(myorder)
                # FIXME: store order with attributes in order pandas
                self.openorders.append(myorder)
                quote_amount -= amount * price
        else:
            print(f"unsupported base {base}")

    def buy_ratio(self, buylist):
        buydict = dict()
        free_distribution = 1/len(buylist)  # equally distributed weight
        # TODO: 50% of weight via Alpha of that currency
        for base in buylist:
            buydict[base] = free_distribution
        return buydict

    def trade_loop(self, cpc, buy_trshld, sell_trshld):
        buylist = list()
        try:
            while True:
                print(f"{env.nowstr()} next round")
                # TOD: check order progress
                ts1 = pd.Timestamp.utcnow()
                for base in Bk.book.index:
                    if base in Xch.black_bases:
                        continue
                    sym = base + "/" + Env.quote
                    ttf = ctf.TargetsFeatures(base)
                    ohlcv_df = Xch.get_ohlcv(base, env.Env.minimum_minute_df_len, datetime.utcnow())
                    # print(sym)
                    if ohlcv_df is None:
                        print(f"{env.nowstr()} removing {base} from book due missing ohlcv")
                        Bk.book = Bk.book.drop([base])
                        continue
                    try:
                        ttf.calc_features_and_targets(ohlcv_df)
                    except env.MissingHistoryData as err:
                        print(f"{env.nowstr()} removing {base} from book due to error: {err}")
                        Bk.book = Bk.book.drop([base])
                        continue
                    tfv = ttf.vec.iloc[[len(ttf.vec)-1]]
                    cl = cpc.performance_with_features(tfv, buy_trshld, sell_trshld)
                    # cl will be HOLD if insufficient data history is available

                    # if (base == "ONT") and (cl == ctf.TARGETS[ctf.HOLD]):  # test purposes
                    #     cl = ctf.TARGETS[ctf.SELL]
                    if cl != ctf.TARGETS[ctf.BUY]:
                        # emergency sell in case no SELL signal but performance drops
                        if Bk.book.loc[base, "buyprice"] > 0:
                            pricenow = ohlcv_df.loc[ohlcv_df.index[len(ohlcv_df.index)-1], "close"]
                            if Bk.book.loc[base, "buyprice"] > pricenow:
                                Bk.book.loc[base, "minbelow"] += 1  # minutes below buy price
                            else:
                                Bk.book.loc[base, "minbelow"] = 0  # reset minute counter
                            if Bk.book.loc[base, "minbelow"] > MAX_MINBELOW:  # minutes
                                print("{} Selling {} due to {} minutes < buy price of {}".format(
                                      env.nowstr(), sym,
                                      Bk.book.loc[base, "minbelow"],
                                      Bk.book.loc[base, "buyprice"]))
                                cl = ctf.TARGETS[ctf.SELL]
                                Bk.book.loc[base, "buyprice"] = 0  # reset price monitoring
                                Bk.book.loc[base, "minbelow"] = 0  # minutes below buy price
                    if cl != ctf.TARGETS[ctf.HOLD]:
                        print(f"{env.nowstr()} {base} {ctf.TARGET_NAMES[cl]}")
                    if cl == ctf.TARGETS[ctf.SELL]:
                        self.sell_order(base, ratio=1)
                    if cl == ctf.TARGETS[ctf.BUY]:
                        buylist.append(base)  # amount to be determined by buy_ratio()

                if len(buylist) > 0:
                    buydict = self.buy_ratio(buylist)
                    for base in buydict:
                        self.buy_order(base, ratio=buydict[base])
                    buydict = None
                    buylist.clear()

                ts2 = pd.Timestamp.utcnow()
                tsdiff = 59 - int((ts2 - ts1) / timedelta(seconds=1))  # 1 seconds order progress
                if tsdiff > 1:
                    time.sleep(tsdiff)
                self.check_order_progress()
        except KeyboardInterrupt:
            Bk.safe_cache()
            print(Bk.actions)
            print("finish as requested by keyboard interrupt")


def trading_main():
    print("executing trading_test")
    # env.set_environment(env.Usage.test, env.Calc.ubuntu)
    tee = env.Tee()
    trading = Trading()
    load_classifier = "MLP-ti1-l160-h0.8-l3False-do0.8-optadam_21"
    save_classifier = None
    if True:
        cpc = ck.Cpc(load_classifier, save_classifier)
        cpc.load()
    else:  # repair pickle file
        # from classify_keras import PerfMatrix, EvalPerf

        print("trading pickle repair")
        cpc = ck.Cpc(load_classifier, load_classifier)
        cpc.load()
        classifier = cpc.classifier
        cpc.classifier = None  # don't save TF file again - it does not need repair
        cpc.save()
        cpc.classifier = classifier

    start_time = timeit.default_timer()
    buy_trshld = 0.7
    sell_trshld = 0.7

    # trd.buy_order("ETH", ratio=22/trd.book.loc[Env.quote, "free"])
    # trd.sell_order("ETH", ratio=1)
    trading.trade_loop(cpc, buy_trshld, sell_trshld)
    trading = None  # should trigger Trading destructor

    tdiff = (timeit.default_timer() - start_time) / (60*60)
    print(f"total time: {tdiff:.2f} hours")
    tee.close()


if __name__ == "__main__":
    trading_main()
