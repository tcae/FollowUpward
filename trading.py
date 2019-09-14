#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylama:{name1}={value1}:{name2}={value2}:...
"""
Created on Fri Apr 12 00:26:23 2019

@author: tc
"""
import ccxt
import pandas as pd
import json
import os
import timeit

import math
# import logging
from datetime import datetime, timedelta  # , timezone
import time
import crypto_targets_features as ctf
import classify_keras as ck
from classify_keras import PerfMatrix, EvalPerf

CACHE_PATH = f"{ctf.DATA_PATH_PREFIX}cache/"
RETRIES = 5  # number of ccxt retry attempts before proceeding without success
MAX_MINBELOW = 0  # max minutes below buy price before emergency sell
ORDER_TIMEOUT = 45  # in seconds
MIN_AVG_USDT = 1500  # minimum average minute volume in USDT to be considered
TRADE_VOL_LIMIT_USDT = 100
BASES = ["BTC", "XRP", "ETH", "BNB", "EOS", "LTC", "NEO", "TRX", "USDT"]
BLOCKED_ASSET_AMOUNT = {"BNB": 100, "USDT": 20000}  # amounts in base currency
# BASES = ["USDT"]
BLACK_BASES = ["TUSD", "USDT", "ONG", "PAX", "BTT", "ATOM", "FET", "USDC", "ONE", "CELR", "LINK"]
QUOTE = "USDT"
DATA_KEYS = ["open", "high", "low", "close", "volume"]
AUTH_FILE = ctf.HOME  + ".catalyst/data/exchanges/binance/auth.json"
MPH = 60  # Minutes Per Hour
ICEBERG_USDT_PART = 450
MAX_USDT_ORDER = ICEBERG_USDT_PART * 10  # binanace limitP10 iceberg parts per order
# logging.basicConfig(level=logging.DEBUG)


def nowstr():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


class Xch():
    """Encapsulation of ccxt and tradable cryptos and asset portfolio that is linked to
    tradable cryptos. That reduces borderplate complexity when using ccxt in this specfic
     context.


    """

    def __init__(self):
        fname = AUTH_FILE
        if not os.path.isfile(AUTH_FILE):
            print(f"file {AUTH_FILE} not found")
            return
        try:
            with open(fname, "rb") as fp:
                self.auth = json.load(fp)
            if ("name" not in self.auth) or ("key" not in self.auth) or \
                    ("secret" not in self.auth):
                print(f"missing auth keys: {self.auth}")
        except IOError:
            print(f"IO error while trying to open {fname}")
            return
        self.xch = ccxt.binance({
            "apiKey": self.auth["key"],
            "secret": self.auth["secret"],
            "timeout": 10000,
            "enableRateLimit": True,
        })
        assert self.xch.has["fetchTickers"]
        assert self.xch.has["fetchOpenOrders"]
        assert self.xch.has["fetchOHLCV"]
        assert self.xch.has["fetchBalance"]
        assert self.xch.has["fetchTickers"]
        assert self.xch.has["createLimitOrder"]
        assert self.xch.has["cancelOrder"]
        assert self.xch.has["fetchOrderBook"]
        assert self.xch.has["fetchTrades"]
        assert self.xch.has["fetchMyTrades"]
        # kraken in phase 2
        # kraken = ccxt.kraken({
        #     "apiKey": "YOUR_PUBLIC_API_KEY",
        #     "secret": "YOUR_SECRET_PRIVATE_KEY",
        # })

        self.markets = self.xch.load_markets()
        self.actions = pd.DataFrame(columns=["ccxt", "timestamp"])  # action_id is the index nbr

    def load_markets(self):
        return self.xch.load_markets()

    def fetch_balance(self):
        return self.xch.fetch_balance()

    def public_get_exchangeinfo(self):
        return self.xch.public_get_exchangeinfo()

    def update_bookkeeping(self, tickers):
        xch_info = self.xch.public_get_exchangeinfo()
        mybal = self.myfetch_balance("update_bookkeeping")
        if mybal is None:
            return
        self.log_action("fetch_balance (update_bookkeeping)")
        bases = BASES.copy()
        self.book = pd.DataFrame(index=bases, columns=["action_id", "free", "used", "USDT",
                                                       "dayUSDT", "updated"])
        for base in mybal:
            sym = base + "/" + QUOTE
            if (base not in bases) and (base not in ["free", "info", "total", "used"]):
                if sym in tickers:
                    bval = (mybal[base]["free"] + mybal[base]["used"]) * tickers[sym]["last"]
                    if bval > 1:
                        bases.append(base)
                else:
                    bval = mybal[base]["free"] + mybal[base]["used"]
                    if bval > 0.01:  # whatever USDT value
                        #  print(f"balance value of {bval} {base} not traded in USDT")
                        pass
        for sym in tickers:
            if sym.endswith("/USDT"):  # add markets above MIN_AVG_USDT*60*24 daily volume
                base = sym[:-5]
                if base not in BLACK_BASES:
                    if tickers[sym]["quoteVolume"] >= (MIN_AVG_USDT*60*24):
                        bases.append(base)
        for base in bases:
            self.update_book_entry(base, mybal, tickers)
            self.book.loc[base, "buyprice"] = 0  # indicating no open buy
            self.book.loc[base, "minbelow"] = 0  # minutes below buy price
            syms = xch_info["symbols"]
            for s in syms:
                if (s["baseAsset"] == base) and (s["quoteAsset"] == QUOTE):
                    for f in s["filters"]:
                        if f["filterType"] == "ICEBERG_PARTS":
                            self.book.loc[base, "iceberg_parts"] = int(f["limit"])
                        if f["filterType"] == "LOT_SIZE":
                            self.book.loc[base, "lot_size_min"] = float(f["minQty"])
                            assert f["minQty"] == f["stepSize"]

    def update_balance(self, tickers):
        mybal = self.xch.myfetch_balance("update_balance")
        if mybal is None:
            return
        self.log_action("fetch_balance (update_balance)")
        for base in self.book.index:
            assert base in mybal, f"unexpectedly missing {base} in balance"
            self.update_book_entry(base, mybal, tickers)

    def fetch_ohlcv(self, sym, timeframe, since, limit):
        # return self.xch.fetch_ohlcv(sym, timeframe, since=since, limit=limit)
        ohlcvs = None
        for i in range(RETRIES):
            try:
                ohlcvs = self.xch.fetch_ohlcv(sym, "1m", since=since, limit=limit)
                # only 1000 entries are returned by one fetch
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} fetch_ohlcv failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            except ccxt.DDoSProtection as err:
                print(f"{nowstr()} fetch_ohlcv failed {i}x due to a DDoSProtection error:",
                      str(err))
                continue
            except ccxt.ExchangeNotAvailable as err:
                print(f"{nowstr()} fetch_ohlcv failed {i}x due to a ExchangeNotAvl error:",
                      str(err))
                continue
            except ccxt.InvalidNonce as err:
                print(f"{nowstr()} fetch_ohlcv failed {i}x due to a InvalidNonce error:",
                      str(err))
                continue
            except ccxt.NetworkError as err:
                print(f"{nowstr()} fetch_ohlcv failed {i}x due to a NetworkError error:",
                      str(err))
                continue
            except ccxt.ArgumentsRequired as err:
                print(f"{nowstr()} fetch_ohlcv failed {i}x due to a ArgumentsReqd error:",
                      str(err))
                break
            except ccxt.BadRequest as err:
                print(f"{nowstr()} fetch_ohlcv failed {i}x due to a BadRequest error:",
                      str(err))
                break
            except ccxt.NullResponse as err:
                print(f"{nowstr()} fetch_ohlcv failed {i}x due to a NullResponse error:",
                      str(err))
                break
            except ccxt.BadResponse as err:
                print(f"{nowstr()} fetch_ohlcv failed {i}x due to a BadResponse error:",
                      str(err))
                break
            except ccxt.AddressPending as err:
                print(f"{nowstr()} fetch_ohlcv failed {i}x due to a AddressPending error:",
                      str(err))
                break
            except ccxt.InvalidAddress as err:
                print(f"{nowstr()} fetch_ohlcv failed {i}x due to a InvalidAddress error:",
                      str(err))
                break
            except ccxt.NotSupported as err:
                print(f"{nowstr()} fetch_ohlcv failed {i}x due to a NotSupported error:",
                      str(err))
                break
            except ccxt.ExchangeError as err:
                print(f"{nowstr()} fetch_ohlcv failed {i}x due to a ExchangeError error:",
                      str(err))
                break
            else:
                break
        if ohlcvs is None:
            print(f"{nowstr()} get_ohlcv ERROR: cannot fetch_ohlcv")
            return None
        if len(ohlcvs) == 0:
            print(f"{nowstr()} get_ohlcv ERROR: empty fetch_ohlcv {since} {limit}")
            return None
        self.log_action("fetch_ohlcv")
        return ohlcvs

    def fetch_open_orders(self, sym):
        return self.xch.fetch_open_orders(sym)

    def cancel_order(self, orderid, sym):
        return self.xch.cancel_order(orderid, sym)

    def fetch_tickers(self):
        return self.xch.fetch_tickers()

    def create_limit_sell_order(self, sym, amount, price, *params):
        # return self.xch.create_limit_sell_order(sym, amount, price, *params)
        myorder = None
        for i in range(RETRIES):
            try:
                myorder = self.xch.create_limit_sell_order(sym, amount, price, *params)
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} sell_order failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            else:
                break
        if myorder is None:
            print(f"nowstr() sell_order ERROR: cannot create_limit_sell_order")
        else:
            self.log_action("create_limit_sell_order")
        return myorder

    def create_limit_buy_order(self, base, amount, price, *params):
        # return self.xch.create_limit_buy_order(sym, amount, price, *params)
        sym = base + "/" + QUOTE
        myorder = None
        for i in range(RETRIES):
            try:
                myorder = self.xch.create_limit_buy_order(sym, amount, price, *params)
            except ccxt.InsufficientFunds as err:
                print(f"{nowstr()} buy_order failed due to a InsufficientFunds ",
                      str(err))
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} buy_order failed {i}x due to a RequestTimeout error: ",
                      str(err))
                continue
            else:
                break
        if myorder is None:
            print(f"nowstr() buy_order ERROR: cannot create_limit_buy_order")
            return
        if self.book.loc[base, "buyprice"] < price:
            self.book.loc[base, "buyprice"] = price
        self.book.loc[base, "minbelow"] = 0  # minutes below buy price
        self.log_action("create_limit_buy_order")

    def myfetch_balance(self, caller):
        mybal = None
        for i in range(RETRIES):
            try:
                mybal = self.xch.fetch_balance()
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} fetch_balance failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            else:
                self.log_action(f"fetch_balance ({caller})")
                break
        if mybal is None:
            print(f"nowstr() {caller} ERROR: cannot fetch_balance")
        return mybal

    def fetch_trades(self, sym):
        trades = None
        for i in range(RETRIES):
            try:
                trades = self.xch.fetch_trades(sym)  # returns 500 trades
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} fetch_trades failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            else:
                break
        if trades is None:
            print(f"nowstr() log_trades_orderbook ERROR: cannot fetch_trades")
            return
        self.log_action("fetch_trades")

    def log_action(self, ccxt_action):
        """
        self.aid is an action_id to log the sequence of actions
        self.ats is an action_id timestamp to log the sequence of actions
        """
        self.ats = nowstr()  # pd.Timestamp.utcnow()
        self.aid = len(self.actions.index)
        self.actions.loc[self.aid] = [ccxt_action, self.ats]

    def log_trades_orderbook(self, base):
        """
        Receives a base and a sell or buy signal.
        According to order strategy free quote will be used to fill.
        """
        if base in BLACK_BASES:
            return
        sym = base + "/" + QUOTE
        ob = None
        for i in range(RETRIES):
            try:
                ob = self.xch.fetch_order_book(sym)  # returns 100 bids and 100 asks
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} fetch_order_book failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            else:
                break
        if ob is None:
            print(f"nowstr() log_trades_orderbook ERROR: cannot fetch_order_book")
            return
        self.log_action("fetch_order_book")
        for pa in ob["bids"]:
            oix = len(self.orderbook.index)
            self.orderbook.loc[oix] = [self.aid, base, sym, self.ats, "bid", pa[0], pa[1]]
        for pa in ob["asks"]:
            oix = len(self.orderbook.index)
            self.orderbook.loc[oix] = [self.aid, base, sym, self.ats, "ask", pa[0], pa[1]]

        trades = self.xch.fetch_trades(sym)
        if trades is None:
            return
        for trade in trades:
            ix = len(self.trades.index)
            ts = pd.to_datetime(datetime.utcfromtimestamp(trade["timestamp"]/1000))
            self.trades.loc[ix] = [self.aid, base, sym, ts,
                                   trade["side"], trade["price"], trade["amount"]]

    def __del__(self):
        print(self.actions)
        print("Trading destructor called, store log pandas.")

    def safe_cache(self):
        for base in self.book.index:
            if base in self.ohlcv:
                df = self.ohlcv[base]
                df = df[DATA_KEYS]
                df = df.drop([df.index[len(df.index)-1]])  # the last candle is incomplete
                ctf.save_asset_dataframe(df, CACHE_PATH, ctf.sym_of_base(base))

    def update_book_entry(self, base, mybal, tickers):
        sym = base + "/" + QUOTE
        if (sym in tickers) or (base == QUOTE):
            self.book.loc[base, "action_id"] = self.aid
            self.book.loc[base, "free"] = mybal[base]["free"]
            self.book.loc[base, "used"] = mybal[base]["used"]
            if base == QUOTE:
                self.book.loc[base, "USDT"] = 1
                self.book.loc[base, "dayUSDT"] = 0
            else:
                self.book.loc[base, "USDT"] = tickers[sym]["last"]
                self.book.loc[base, "dayUSDT"] = tickers[sym]["quoteVolume"]
            self.book.loc[base, "updated"] = pd.Timestamp.utcnow()

    def myfetch_open_orders(self, base, caller):
        oo = None
        sym = base + "/" + QUOTE
        for i in range(RETRIES):
            try:
                oo = self.xch.fetch_open_orders(sym)
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} fetch_open_orders failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            else:
                break
        if oo is None:
            print(f"nowstr() {caller} ERROR: cannot fetch_tickers")
        else:
            self.log_action("fetch_open_orders")
        return oo

    def myfetch_cancel_orders(self, orderid, base):
        sym = base + "/" + QUOTE
        for i in range(RETRIES):
            try:
                self.xch.cancel_order(orderid, sym)
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} cancel_order failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            except ccxt.OrderNotFound:
                break  # order is already closed
            else:
                break
        # for a request to cancelOrder a user is required to retry the same call the second time.
        # If instead of a retry a user calls a fetchOrder, fetchOrders, fetchOpenOrders or
        # fetchClosedOrders right away without a retry to call cancelOrder, this may cause
        # the .orders cache to fall out of sync.
        for i in range(RETRIES):
            try:
                self.xch.cancel_order(orderid, sym)
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} 2nd cancel_order failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            except ccxt.OrderNotFound:
                break  # order is already closed
            else:
                break

    def check_limits(self, base, amount, price):
        sym = base + "/" + QUOTE
        # ice_chunk = ICEBERG_USDT_PART / price # about the chunk quantity we want
        ice_chunk = self.xch.book.loc[base, "dayUSDT"] / (24 * 60 * 4)  # 1/4 of average minute volume
        ice_parts = math.ceil(amount / ice_chunk)  # about equal parts

        mincost = self.markets[sym]["limits"]["cost"]["min"]  # test purposes
        ice_parts = math.floor(price * amount / mincost)  # test purposes

        ice_parts = min(ice_parts, self.xch.book.loc[base, "iceberg_parts"])
        if ice_parts > 1:
            ice_chunk = amount / ice_parts
            ice_chunk = int(ice_chunk / self.xch.book.loc[base, "lot_size_min"]) \
                * self.xch.book.loc[base, "lot_size_min"]
            ice_chunk = round(ice_chunk, self.markets[sym]["precision"]["amount"])
            amount = ice_chunk * ice_parts
            if ice_chunk == amount:
                ice_chunk = 0
        else:
            ice_chunk = 0
        # amount = round(amount, self.markets[sym]["precision"]["amount"]) is ensured by ice_chunk
        price = round(price, self.markets[sym]["precision"]["price"])
        minamount = self.markets[sym]["limits"]["amount"]["min"]
        if minamount is not None:
            if amount <= minamount:
                print(f"{nowstr()} limit violation: {sym} amount {amount} <= min({minamount})")
                return (0, price, 0)
        maxamount = self.markets[sym]["limits"]["amount"]["max"]
        if maxamount is not None:
            if amount >= maxamount:
                print(f"{nowstr()} limit violation: {sym} amount {amount} >= min({maxamount})")
                return (0, price, 0)
        minprice = self.markets[sym]["limits"]["price"]["min"]
        if minprice is not None:
            if price <= minprice:
                print(f"{nowstr()} limit violation: {sym} price {price} <= min({minprice})")
                return (amount, 0, ice_chunk)
        maxprice = self.markets[sym]["limits"]["price"]["max"]
        if maxprice is not None:
            if price >= maxprice:
                print(f"{nowstr()} limit violation: {sym} price {price} >= min({maxprice})")
                return (amount, 0, ice_chunk)
        cost = price * amount
        mincost = self.markets[sym]["limits"]["cost"]["min"]
        if mincost is not None:
            if cost <= mincost:
                print(f"{nowstr()} limit violation: {sym} cost {cost} <= min({mincost})")
                return (0, 0, 0)
        maxcost = self.markets[sym]["limits"]["cost"]["max"]
        if maxcost is not None:
            if cost >= maxcost:
                print(f"{nowstr()} limit violation: {sym} cost {cost} >= min({maxcost})")
                return (0, 0, 0)
        return (amount, price, ice_chunk)


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
        self.xch = Xch()

        # FIXME: store order with attributes in order pandas
        self.openorders = list()
        self.ohlcv = dict()
        # TODO: load orderbook and trades in case they were stored
        self.orderbook = pd.DataFrame(columns=["action_id", "base", "sym", "timestamp",
                                               "side", "price", "amount"])
        self.trades = pd.DataFrame(columns=["action_id", "base", "sym", "timestamp",
                                            "side", "price", "amount"])
        tickers = self.myfetch_tickers("__init__")
        if tickers is not None:
            self.xch.update_bookkeeping(tickers)
            bdf = self.xch.book[["free", "used", "USDT", "dayUSDT"]]
            print(bdf)
            for base in self.xch.book.index:
                adf = None
                if base not in BLACK_BASES:
                    try:
                        adf = ctf.load_asset_dataframe(CACHE_PATH, base.lower())
                    except ctf.MissingHistoryData:
                        pass
                    if adf is not None:
                        self.ohlcv[base] = adf

    def last_hour_performance(self, ohlcv_df):
        cix = ohlcv_df.columns.get_loc("close")
        tix = len(ohlcv_df.index) - 1
        last_close = ohlcv_df.iat[tix, cix]
        tix = max(0, tix - 60)
        hour_close = ohlcv_df.iat[tix, cix]
        perf = (last_close - hour_close)/hour_close
        return perf

    def get_ohlcv(self, base, minutes, when):
        """Returns the last 'minutes' OHLCV values of pair before 'when'.

        Is also used to calculate the average minute volume over the last hour
        that is used in buy orders to determine the max buy volume.

        Exceptions:
            1) gaps in tics (e.g. maintenance)
        ==> will be filled with last values before the gap,
        2) when before df cache coverage (e.g. simulation) ==> # TODO
        """
        when = pd.Timestamp(when).replace(second=0, microsecond=0, nanosecond=0, tzinfo=None)
        when = when.to_pydatetime()
        minutes += 1
        isincheck = True in self.xch.book.index.isin([base])
        df = None
        if isincheck:
            if base in self.ohlcv:
                df = self.ohlcv[base]
                df = df[DATA_KEYS]
                df = df.drop([df.index[len(df.index)-1]])  # because the last candle is incomplete
            sym = base + "/" + QUOTE
            remaining = minutes
            while remaining > 0:
                if df is not None:
                    last_tic = df.index[len(df.index)-1].to_pydatetime()
                    dtlast = last_tic.replace(tzinfo=None)
                    dfdiff = int((when - dtlast) / timedelta(minutes=1))
                    if dfdiff < remaining:
                        remaining = dfdiff
                if remaining <= 0:
                    break
                fromdate = when - timedelta(minutes=remaining)
                since = int((fromdate - datetime(1970, 1, 1)).total_seconds() * 1000)
                # print(f"{nowstr()} {base} fromdate {ctf.timestr(fromdate)} minutes {remaining}")
                ohlcvs = self.xch.fetch_ohlcv(sym, "1m", since=since, limit=remaining)
                # only 1000 entries are returned by one fetch
                if ohlcvs is None:
                    return None
                prev_tic = fromdate - timedelta(minutes=1)
                itic = None
                for ohlcv in ohlcvs:
                    if df is None:
                        df = pd.DataFrame(columns=DATA_KEYS)
                    tic = pd.to_datetime(datetime.utcfromtimestamp(ohlcv[0]/1000))
                    if int((tic - prev_tic)/timedelta(minutes=1)) > 1:
                        print(f"ohlcv time gap for {base} between {prev_tic} and {tic}")
                        if prev_tic < fromdate:  # repair first tics
                            print(f"no repair of missing history data for {base}")
                            return None
                            prev_tic += timedelta(minutes=1)
                            iptic = itic = pd.Timestamp(prev_tic, tz=None)
                            df.loc[iptic] = [ohlcv[1], ohlcv[2], ohlcv[3], ohlcv[4], ohlcv[5]]
                            remaining -= 1
                        prev_tic += timedelta(minutes=1)
                        while (prev_tic < tic):
                            iptic = pd.Timestamp(prev_tic, tz=None)
                            df.loc[iptic] = df.loc[itic]
                            remaining -= 1
                            prev_tic += timedelta(minutes=1)
                    itic = pd.Timestamp(tic, tz=None)
                    df.loc[itic] = [ohlcv[1], ohlcv[2], ohlcv[3], ohlcv[4], ohlcv[5]]
                    remaining -= 1
                    prev_tic += timedelta(minutes=1)

            self.ohlcv[base] = df
            assert len(df) >= minutes, f"{base} len(df) {len(df)} < {minutes} minutes"
            self.xch.book.loc[base, "updated"] = nowstr()
        else:
            print(f"unsupported base {base}")
        return df

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
        for base in self.xch.book.index:
            if self.xch.book.loc[base, "used"] > 0:
                oo = self.myfetch_open_orders(base, "check_order_progress")
                if oo is not None:
                    for order in oo:
                        is_sell = (order["side"] == "sell")
                        ots = pd.Timestamp(order["datetime"])
                        nowts = pd.Timestamp.utcnow()
                        tsdiff = int((nowts - ots) / timedelta(seconds=1))
                        print(f"now {ctf.timestr(nowts)} - ts {ctf.timestr(ots)} = {tsdiff}s")
                        if tsdiff >= ORDER_TIMEOUT:
                            try:
                                self.myfetch_cancel_orders(order["id"], base)
                                self.myfetch_cancel_orders(order["id"], base)
                            except ccxt.NetworkError:
                                self.myfetch_cancel_orders(order["id"], base)
                            except ccxt.OrderNotFound:
                                # that's what we are looking for
                                pass
                            if is_sell:
                                self.sell_order(base, ratio=1)

    def myfetch_tickers(self, caller):
        tickers = None
        for i in range(RETRIES):
            try:
                tickers = self.xch.fetch_tickers()
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} fetch_tickers failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            else:
                break
        if tickers is None:
            print(f"nowstr() {caller} ERROR: cannot fetch_tickers")
        return tickers

    def sell_order(self, base, ratio=1):
        """ Sells the ratio of free base currency. Constraint: 0 <= ratio <= 1
        """
        assert 0 <= ratio <= 1, f"not compliant to constraint: 0 <= ratio {ratio} <= 1"
        isincheck = True in self.xch.book.index.isin([base])
        if isincheck:
            sym = base + "/" + QUOTE
            tickers = self.myfetch_tickers("sell_order")
            if tickers is None:
                return
            self.xch.update_balance(tickers)
            base_amount = self.xch.book.loc[base, "free"]
            if base in BLOCKED_ASSET_AMOUNT:
                base_amount -= BLOCKED_ASSET_AMOUNT[base]
            base_amount *= ratio
            price = tickers[sym]["bid"]  # TODO order spread strategy
            print(f"{nowstr()} SELL {base_amount} {base} x {price} {sym}")
            while base_amount > 0:
                price = tickers[sym]["bid"]  # TODO order spread strategy
                max_chunk = MAX_USDT_ORDER / price
                ice_chunk = amount = min([base_amount, max_chunk])
                (amount, price, ice_chunk) = self.xch.check_limits(base, amount, price)
                if (amount == 0) or (price == 0):
                    return
                myorder = self.xch.create_limit_sell_order(
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
        for base in self.xch.book.index:
            if base not in BLACK_BASES:
                trade_vol += (self.xch.book.loc[base, "free"] + self.xch.book.loc[base, "used"]) \
                             * self.xch.book.loc[base, "USDT"]
        trade_vol = TRADE_VOL_LIMIT_USDT - trade_vol
        trade_vol = min(trade_vol, self.xch.book.loc[QUOTE, "free"])
        if QUOTE in BLOCKED_ASSET_AMOUNT:
            trade_vol -= BLOCKED_ASSET_AMOUNT[QUOTE]
        usdt_amount = trade_vol * ratio
        return usdt_amount

    def buy_order(self, base, ratio=1):
        """ Buys the ratio of free quote currency with base currency. Constraint: 0 <= ratio <= 1
        """
        assert 0 <= ratio <= 1, f"not compliant to constraint: 0 <= ratio {ratio} <= 1"
        isincheck = True in self.xch.book.index.isin([base])
        if isincheck:
            sym = base + "/" + QUOTE
            tickers = self.myfetch_tickers("buy_order")
            if tickers is None:
                return

            self.xch.update_balance(tickers)
            quote_amount = self.trade_amount(base, ratio)
            price = tickers[sym]["ask"]  # TODO order spread strategy
            print(f"{nowstr()} BUY {quote_amount} USDT / {price} {sym}")
            while quote_amount > 0:
                price = tickers[sym]["ask"]  # TODO order spread strategy
                max_chunk = MAX_USDT_ORDER / price
                ice_chunk = amount = min([quote_amount/price, max_chunk])
                (amount, price, ice_chunk) = self.xch.check_limits(base, amount, price)
                if (amount == 0) or (price == 0):
                    return
                myorder = self.xch.create_limit_buy_order(
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
                print(f"{nowstr()} next round")
                # TOD: check order progress
                ts1 = pd.Timestamp.utcnow()
                for base in self.xch.book.index:
                    if base in BLACK_BASES:
                        continue
                    sym = base + "/" + QUOTE
                    ttf = ctf.TargetsFeatures(base, QUOTE)
                    ohlcv_df = self.get_ohlcv(base, ttf.minimum_minute_df_len, datetime.utcnow())
                    # print(sym)
                    if ohlcv_df is None:
                        print(f"{nowstr()} removing {base} from book due missing ohlcv")
                        self.xch.book = self.xch.book.drop([base])
                        continue
                    try:
                        ttf.calc_features_and_targets(ohlcv_df)
                    except ctf.MissingHistoryData as err:
                        print(f"{nowstr()} removing {base} from book due to error: {err}")
                        self.xch.book = self.xch.book.drop([base])
                        continue
                    tfv = ttf.vec.iloc[[len(ttf.vec)-1]]
                    cl = cpc.performance_with_features(tfv, buy_trshld, sell_trshld)
                    # cl will be HOLD if insufficient data history is available

                    # if (base == "ONT") and (cl == ctf.TARGETS[ctf.HOLD]):  # test purposes
                    #     cl = ctf.TARGETS[ctf.SELL]
                    if cl != ctf.TARGETS[ctf.BUY]:
                        # emergency sell in case no SELL signal but performance drops
                        if self.xch.book.loc[base, "buyprice"] > 0:
                            pricenow = ohlcv_df.loc[ohlcv_df.index[len(ohlcv_df.index)-1], "close"]
                            if self.xch.book.loc[base, "buyprice"] > pricenow:
                                self.xch.book.loc[base, "minbelow"] += 1  # minutes below buy price
                            else:
                                self.xch.book.loc[base, "minbelow"] = 0  # reset minute counter
                            if self.xch.book.loc[base, "minbelow"] > MAX_MINBELOW:  # minutes
                                print("{} Selling {} due to {} minutes < buy price of {}".format(
                                      nowstr(), sym,
                                      self.xch.book.loc[base, "minbelow"],
                                      self.xch.book.loc[base, "buyprice"]))
                                cl = ctf.TARGETS[ctf.SELL]
                                self.xch.book.loc[base, "buyprice"] = 0  # reset price monitoring
                                self.xch.book.loc[base, "minbelow"] = 0  # minutes below buy price
                    if cl != ctf.TARGETS[ctf.HOLD]:
                        print(f"{nowstr()} {base} {ctf.TARGET_NAMES[cl]}")
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
            self.xch.safe_cache()
            print("finish as requested by keyboard interrupt")

    def show_all_binance_commands(self):
        binance_api = dir(ccxt.binance())
        for cmd in binance_api:
            print(cmd)

    def show_binance_base_constraints(self, base):
        dct = self.xch.public_get_exchangeinfo()
        syms = dct["symbols"]
        for s in syms:
            if (s["baseAsset"] == base) and (s["quoteAsset"] == "USDT"):
                for f in s["filters"]:
                    print(f)


if __name__ == "__main__":
    tee = ctf.Tee(f"{ck.MODEL_PATH}Log_{ctf.timestr()}.txt")
    trd = Trading()
    load_classifier = "MLP-ti1-l160-h0.8-l3False-do0.8-optadam_21"
    save_classifier = None
    cpc = ck.Cpc(load_classifier, save_classifier)
    cpc.load()

    start_time = timeit.default_timer()
    buy_trshld = 0.7
    sell_trshld = 0.7

    # trd.buy_order("ETH", ratio=22/trd.xch.book.loc[QUOTE, "free"])
    # trd.sell_order("ETH", ratio=1)
    trd.trade_loop(cpc, buy_trshld, sell_trshld)
    trd = None  # should trigger Trading destructor

    tdiff = (timeit.default_timer() - start_time) / (60*60)
    print(f"total time: {tdiff:.2f} hours")
    tee.close()
