#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import targets_features as tf
from crypto_classification import Cpc
import crypto_classification as cc

CACHE_PATH = '/Users/tc/crypto/cache'
RETRIES = 5  # number of ccxt retry attempts before proceeding without success
MAX_MINBELOW = 0  # max minutes below buy price before emergency sell
ORDER_TIMEOUT = 45  # in seconds
MIN_AVG_USDT = 1500  # minimum average minute volume in USDT to be considered
TRADE_VOL_LIMIT_USDT = 100
# BASES = ['BTC', 'XRP', 'ETH', 'BNB', 'EOS', 'LTC', 'NEO', 'TRX', 'USDT']
BASES = ['USDT']
BLACK_BASES = ['TUSD', 'USDT', 'BNB', 'ONG', 'PAX', 'BTT', 'ATOM', 'FET', 'USDC']
QUOTE = 'USDT'
DATA_KEYS = ['open', 'high', 'low', 'close', 'volume']
AUTH_FILE = "/Users/tc/.catalyst/data/exchanges/binance/auth.json"
MPH = 60  # Minutes Per Hour
ICEBERG_USDT_PART = 450
MAX_USDT_ORDER = ICEBERG_USDT_PART * 10  # binanace limit is 10 iceberg parts per order
# logging.basicConfig(level=logging.DEBUG)

def nowstr():
    return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

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
        fname = AUTH_FILE
        if not os.path.isfile(AUTH_FILE):
            print(f"file {AUTH_FILE} not found")
            return
        try:
            with open(fname, 'rb') as fp:
                self.auth = json.load(fp)
            if ('name' not in self.auth) or ('key' not in self.auth) or \
                ('secret' not in self.auth):
                print(f"missing auth keys: {self.auth}")
        except IOError:
            print(f"IO error while trying to open {fname}")
            return
        self.xch  = ccxt.binance({
            'apiKey': self.auth['key'],
            'secret': self.auth['secret'],
            'timeout': 10000,
            'enableRateLimit': True,
        })
        assert self.xch.has['fetchTickers']
        assert self.xch.has['fetchOpenOrders']
        assert self.xch.has['fetchOHLCV']
        assert self.xch.has['fetchBalance']
        assert self.xch.has['fetchTickers']
        assert self.xch.has['createLimitOrder']
        assert self.xch.has['cancelOrder']
        assert self.xch.has['fetchOrderBook']
        assert self.xch.has['fetchTrades']
        assert self.xch.has['fetchMyTrades']
        # kraken in phase 2
        # kraken = ccxt.kraken({
        #     'apiKey': 'YOUR_PUBLIC_API_KEY',
        #     'secret': 'YOUR_SECRET_PRIVATE_KEY',
        # })

        self.markets = self.xch.load_markets()

        # FIXME: store order with attributes in order pandas
        self.openorders = list()
        self.ohlcv = dict()
        # TODO: load orderbook and trades in case they were stored
        self.actions = pd.DataFrame(columns=['ccxt', 'timestamp'])  # action_id is the index nbr
        self.orderbook = pd.DataFrame(columns=['action_id', 'base', 'sym', 'timestamp',
                                               'side', 'price', 'amount'])
        self.trades = pd.DataFrame(columns=['action_id', 'base', 'sym', 'timestamp',
                                            'side', 'price', 'amount'])
        tickers = self.myfetch_tickers("__init__")
        if tickers is not None:
            self.update_bookkeeping(tickers)
            df = self.book[['free', 'used', 'USDT', 'dayUSDT']]
            print(df)
            for base in self.book.index:
                if base not in BLACK_BASES:
                    sym = base + '/' + QUOTE
                    df = tf.load_asset_dataframe(CACHE_PATH, sym)
                    if df is not None:
                        self.ohlcv[base] = df

    def __del__(self):
        print(self.actions)
        print('Trading destructor called, store log pandas.')

    def safe_cache(self):
        for base in self.book.index:
            isincheck = True in self.book.index.isin([base])
            df = None
            if isincheck and (base not in BLACK_BASES):
                if base in self.ohlcv:
                    df = self.ohlcv[base]
                    df = df[DATA_KEYS]
                    df = df.drop([df.index[len(df.index)-1]])  # because the last candle is incomplete
                    sym = base + '/' + QUOTE
                    tf.save_asset_dataframe(df, CACHE_PATH, sym)

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


    def update_book_entry(self, base, mybal, tickers):
        sym = base + '/' + QUOTE
        if (sym in tickers) or (base == QUOTE):
            self.book.loc[base, 'action_id'] = self.aid
            self.book.loc[base, 'free'] = mybal[base]['free']
            self.book.loc[base, 'used'] = mybal[base]['used']
            if base == QUOTE:
                self.book.loc[base, 'USDT'] = 1
            else:
                self.book.loc[base, 'USDT'] = tickers[sym]['last']
                self.book.loc[base, 'dayUSDT'] = tickers[sym]['quoteVolume']
            self.book.loc[base, 'updated'] = pd.Timestamp.utcnow()

    def update_bookkeeping(self, tickers):
        xch_info = self.xch.public_get_exchangeinfo()
        mybal = self.myfetch_balance("update_bookkeeping")
        if mybal is None:
            return
        self.log_action('fetch_balance (update_bookkeeping)')
        bases = BASES.copy()
        self.book = pd.DataFrame(index=bases, columns=['action_id', 'free', 'used', 'USDT',
                                                       'dayUSDT', 'updated'])
        for base in mybal:
            sym = base + '/' + QUOTE
            if (base not in bases) and (base not in ['free', 'info', 'total', 'used']):
                if sym in tickers:
                    bval = (mybal[base]['free'] + mybal[base]['used']) * tickers[sym]['last']
                    if bval > 1:
                        bases.append(base)
                else:
                    bval = mybal[base]['free'] + mybal[base]['used']
                    if bval > 0.01:  # whatever USDT value
                        #  print(f"balance value of {bval} {base} not traded in USDT")
                        pass
        for sym in tickers:
            if sym.endswith('/USDT'):  # add markets above MIN_AVG_USDT*60*24 daily volume
                base = sym[:-5]
                if base not in BLACK_BASES:
                    if tickers[sym]['quoteVolume'] >= (MIN_AVG_USDT*60*24):
                        bases.append(base)
        for base in bases:
            self.update_book_entry(base, mybal, tickers)
            self.book.loc[base, 'buyprice'] = 0  # indicating no open buy
            self.book.loc[base, 'minbelow'] = 0  # minutes below buy price
            syms = xch_info['symbols']
            for s in syms:
                if (s['baseAsset'] == base) and (s['quoteAsset'] == QUOTE):
                    for f in s['filters']:
                        if f['filterType'] == 'ICEBERG_PARTS':
                            self.book.loc[base, 'iceberg_parts'] = int(f['limit'])
                        if f['filterType'] == 'LOT_SIZE':
                            self.book.loc[base, 'lot_size_min'] = float(f['minQty'])
                            assert f['minQty'] == f['stepSize']

    def update_balance(self, tickers):
        mybal = self.myfetch_balance("update_balance")
        if mybal is None:
            return
        self.log_action('fetch_balance (update_balance)')
        for base in self.book.index:
            assert base in mybal, f"unexpectedly missing {base} in balance"
            self.update_book_entry(base, mybal, tickers)

    def last_hour_performance(self, ohlcv_df):
        cix = ohlcv_df.columns.get_loc('close')
        tix = len(ohlcv_df.index) - 1
        last_close = ohlcv_df.iat[tix, cix]
        tix = max(0, tix - 60)
        hour_close = ohlcv_df.iat[tix, cix]
        perf = (last_close - hour_close)/hour_close
        return perf

    def get_ohlcv(self, base, minutes):
        """Returns the last 'minutes' OHLCV values of pair.

        Is also used to calculate the average minute volume over the last hour
        that is used in buy orders to determine the max buy volume.

        """
        isincheck = True in self.book.index.isin([base])
        df = None
        if isincheck and (base not in BLACK_BASES):
            if base in self.ohlcv:
                df = self.ohlcv[base]
                df = df[DATA_KEYS]
                df = df.drop([df.index[len(df.index)-1]])  # because the last candle is incomplete
            sym = base + '/' + QUOTE
            while minutes > 0:
                dtnow = datetime.utcnow()
                if df is not None:
                    last_tic = df.index[len(df.index)-1].to_pydatetime()
                    dtlast = last_tic.replace(tzinfo=None)
                    dfdiff = int((dtnow - dtlast) / timedelta(minutes=1))
                    if dfdiff < minutes:
                        minutes = dfdiff
                tdminutes = timedelta(minutes=minutes)
                minutes = int(tdminutes / timedelta(minutes=1))
                if (minutes <=0):
                    continue
                fromdate = dtnow - tdminutes
                since = int((fromdate - datetime(1970, 1, 1)).total_seconds() * 1000)
                # print(f"{nowstr()} {sym} fromdate {tf.timestr(fromdate)} minutes {minutes}")
                ohlcvs = None
                for i in range(RETRIES):
                    try:
                        ohlcvs = self.xch.fetch_ohlcv(sym, '1m', since=since, limit=minutes)
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
                        print(f"{nowstr()} fetch_ohlcv failed {i}x due to a ExchangeNotAvl. error:",
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
                    # print(f"{nowstr()} get_ohlcv ERROR: empty fetch_ohlcv {since} {minutes}")
                    # print("{} get_ohlcv ERROR: empty fetch_ohlcv {} {} {}".format(
                    #         nowstr(), fromdate, dtnow, tdminutes))
                    return None
                self.log_action('fetch_ohlcv')
                for ohlcv in ohlcvs:
                    tic = pd.to_datetime(datetime.utcfromtimestamp(ohlcv[0]/1000))
                    minutes -= 1
                    if df is None:
                        df = pd.DataFrame(columns=DATA_KEYS)
                    df.loc[tic] = [ohlcv[1], ohlcv[2], ohlcv[3], ohlcv[4], ohlcv[5]]

            self.ohlcv[base] = df
            self.book.loc[base, 'updated'] = tic
        else:
            print(f"unsupported base {base}")
        return df

    def log_action(self, ccxt_action):
        """
        self.aid is an action_id to log the sequence of actions
        self.ats is an action_id timestamp to log the sequence of actions
        """
        self.ats = pd.Timestamp.utcnow()
        self.aid = len(self.actions.index)
        self.actions.loc[self.aid] = [ccxt_action, self.ats]

    def log_trades_orderbook(self, base):
        """
        Receives a base and a sell or buy signal.
        According to order strategy free quote will be used to fill.
        """
        if base in BLACK_BASES:
            return
        sym = base + '/' + QUOTE
        ob = None
        for i in range(RETRIES):
            try:
                ob = trd.xch.fetch_order_book(sym)  # returns 100 bids and 100 asks
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} fetch_order_book failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            else:
                break
        if ob is None:
            print(f"nowstr() log_trades_orderbook ERROR: cannot fetch_order_book")
            return
        self.log_action('fetch_order_book')
        for pa in ob['bids']:
            oix = len(self.orderbook.index)
            self.orderbook.loc[oix] = [self.aid, base, sym, self.ats, 'bid', pa[0], pa[1]]
        for pa in ob['asks']:
            oix = len(self.orderbook.index)
            self.orderbook.loc[oix] = [self.aid, base, sym, self.ats, 'ask', pa[0], pa[1]]

        trades = None
        for i in range(RETRIES):
            try:
                trades = trd.xch.fetch_trades(sym)  # returns 500 trades
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} fetch_trades failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            else:
                break
        if trades is None:
            print(f"nowstr() log_trades_orderbook ERROR: cannot fetch_trades")
            return
        self.log_action('fetch_trades')
        for trade in trades:
            ix = len(self.trades.index)
            ts = pd.to_datetime(datetime.utcfromtimestamp(trade['timestamp']/1000))
            self.trades.loc[ix] = [self.aid, base, sym, ts,
                                   trade['side'], trade['price'], trade['amount']]

    def myfetch_open_orders(self, base, caller):
        oo = None
        sym = base + '/' + QUOTE
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
        return oo

    def myfetch_cancel_orders(self, orderid, base):
        sym = base + '/' + QUOTE
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
        for base in self.book.index:
            if (self.book.loc[base, 'used'] > 0) and (base not in BLACK_BASES):
                oo = self.myfetch_open_orders(base, 'check_order_progress')
                if oo is not None:
                    self.log_action('fetch_open_orders')
                    for order in oo:
                        is_sell = (order['side'] == 'sell')
                        ots = pd.Timestamp(order['datetime'])
                        nowts = pd.Timestamp.utcnow()
                        tsdiff = int((nowts - ots) / timedelta(seconds=1))
                        print(f"now {tf.timestr(nowts)} - ts {tf.timestr(ots)} = {tsdiff}s")
                        if tsdiff >= ORDER_TIMEOUT:
                            try:
                                self.myfetch_cancel_orders(order['id'], base)
                                self.myfetch_cancel_orders(order['id'], base)
                            except ccxt.NetworkError:
                                self.myfetch_cancel_orders(order['id'], base)
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

    def check_limits(self, base, amount, price):
        sym = base + '/' + QUOTE
        # ice_chunk = ICEBERG_USDT_PART / price # about the chunk quantity we want
        ice_chunk = self.book.loc[base, 'dayUSDT'] / (24 * 60 * 4)  # 1/4 of average minute volume
        ice_parts = math.ceil(amount / ice_chunk) # about equal parts

        mincost = self.markets[sym]['limits']['cost']['min'] # test purposes
        ice_parts = math.floor(price * amount / mincost) # test purposes

        ice_parts = min(ice_parts, self.book.loc[base, 'iceberg_parts'])
        if ice_parts > 1:
            ice_chunk = amount / ice_parts
            ice_chunk = int(ice_chunk / self.book.loc[base, 'lot_size_min']) \
                        * self.book.loc[base, 'lot_size_min']
            ice_chunk = round(ice_chunk, self.markets[sym]['precision']['amount'])
            amount = ice_chunk * ice_parts
            if ice_chunk == amount:
                ice_chunk = 0
        else:
            ice_chunk = 0
        # amount = round(amount, self.markets[sym]['precision']['amount']) is ensured by ice_chunk
        price = round(price, self.markets[sym]['precision']['price'])
        minamount = self.markets[sym]['limits']['amount']['min']
        if minamount is not None:
            if amount <= minamount:
                print(f"{nowstr()} limit violation: {sym} amount {amount} <= min({minamount})")
                return (0, price, 0)
        maxamount = self.markets[sym]['limits']['amount']['max']
        if maxamount is not None:
            if amount >= maxamount:
                print(f"{nowstr()} limit violation: {sym} amount {amount} >= min({maxamount})")
                return (0, price, 0)
        minprice = self.markets[sym]['limits']['price']['min']
        if minprice is not None:
            if price <= minprice:
                print(f"{nowstr()} limit violation: {sym} price {price} <= min({minprice})")
                return (amount, 0, ice_chunk)
        maxprice = self.markets[sym]['limits']['price']['max']
        if maxprice is not None:
            if price >= maxprice:
                print(f"{nowstr()} limit violation: {sym} price {price} >= min({maxprice})")
                return (amount, 0, ice_chunk)
        cost = price * amount
        mincost = self.markets[sym]['limits']['cost']['min']
        if mincost is not None:
            if cost <= mincost:
                print(f"{nowstr()} limit violation: {sym} cost {cost} <= min({mincost})")
                return (0, 0, 0)
        maxcost = self.markets[sym]['limits']['cost']['max']
        if maxcost is not None:
            if cost >= maxcost:
                print(f"{nowstr()} limit violation: {sym} cost {cost} >= min({maxcost})")
                return (0, 0, 0)
        return (amount, price, ice_chunk)

    def sell_order(self, base, ratio=1):
        """ Sells the ratio of free base currency. Constraint: 0 <= ratio <= 1
        """
        assert 0 <= ratio <= 1, f"not compliant to constraint: 0 <= ratio {ratio} <= 1"
        isincheck = True in self.book.index.isin([base])
        if isincheck:
            sym = base + '/' + QUOTE
            tickers = self.myfetch_tickers("sell_order")
            if tickers is None:
                return
            self.update_balance(tickers)
            base_amount = self.book.loc[base, 'free'] * ratio
            price = tickers[sym]['bid']  # TODO order spread strategy
            print(f"{nowstr()} SELL {base_amount} {base} x {price} {sym}")
            while base_amount > 0:
                price = tickers[sym]['bid']  # TODO order spread strategy
                max_chunk = MAX_USDT_ORDER / price
                ice_chunk = amount = min([base_amount, max_chunk])
                (amount, price, ice_chunk) = self.check_limits(base, amount, price)
                if (amount == 0) or (price == 0):
                    return
                myorder = None
                for i in range(RETRIES):
                    try:
                        myorder = self.xch.create_limit_sell_order(sym, amount, price,
                                           {"icebergQty":ice_chunk, "timeInForce": "GTC"})
                    except ccxt.RequestTimeout as err:
                        print(f"{nowstr()} sell_order failed {i}x due to a RequestTimeout error:",
                              str(err))
                        continue
                    else:
                        break
                if myorder is None:
                    print(f"nowstr() sell_order ERROR: cannot create_limit_sell_order")
                    return
                self.log_action('create_limit_sell_order')
                print(myorder)
                # FIXME: store order with attributes in order pandas
                self.openorders.append(myorder)
                base_amount -= amount
        else:
            print(f"unsupported base {base}")

    def trade_amount(self, base, ratio):
        trade_vol = 0
        for base in self.book.index:
            if base not in BLACK_BASES:
                trade_vol += (self.book.loc[base, 'free'] + self.book.loc[base, 'used']) \
                             * self.book.loc[base, 'USDT']
        trade_vol = TRADE_VOL_LIMIT_USDT - trade_vol
        trade_vol = min(trade_vol, self.book.loc[QUOTE, 'free'])
        usdt_amount = trade_vol * ratio
        return usdt_amount

    def buy_order(self, base, ratio=1):
        """ Buys the ratio of free quote currency with base currency. Constraint: 0 <= ratio <= 1
        """
        assert 0 <= ratio <= 1, f"not compliant to constraint: 0 <= ratio {ratio} <= 1"
        isincheck = True in self.book.index.isin([base])
        if isincheck:
            sym = base + '/' + QUOTE
            tickers = self.myfetch_tickers("buy_order")
            if tickers is None:
                return

            self.update_balance(tickers)
            quote_amount = self.trade_amount(base, ratio)
            price = tickers[sym]['ask']  # TODO order spread strategy
            print(f"{nowstr()} BUY {quote_amount} USDT / {price} {sym}")
            while quote_amount > 0:
                price = tickers[sym]['ask']  # TODO order spread strategy
                max_chunk = MAX_USDT_ORDER / price
                ice_chunk = amount = min([quote_amount/price, max_chunk])
                (amount, price, ice_chunk) = self.check_limits(base, amount, price)
                if (amount == 0) or (price == 0):
                    return
                myorder = None
                for i in range(RETRIES):
                    try:
                        myorder = self.xch.create_limit_buy_order(sym, amount, price,
                                          {'icebergQty':ice_chunk, "timeInForce": "GTC"})
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
                if self.book.loc[base, 'buyprice'] < price:
                    self.book.loc[base, 'buyprice'] = price
                self.book.loc[base, 'minbelow'] = 0  # minutes below buy price
                self.log_action('create_limit_buy_order')
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


    def trade_loop(self, cpc, time_aggs, buy_trshld, sell_trshld):
        buylist = list()
        try:
            while True:
                print(f"{nowstr()} next round")
                # TOD: check order progress
                ts1 = pd.Timestamp.utcnow()
                for base in self.book.index:
                    if base in BLACK_BASES:
                        continue
                    sym = base + '/' + QUOTE
                    ohlcv_df = self.get_ohlcv(base, 24*60*10+1)
                    # print(sym)
                    if ohlcv_df is None:
                        continue
                    try:
                        ttf = tf.TargetsFeatures(aggregations=time_aggs, target_key=5, cur_pair=sym)
                        ttf.calc_features_and_targets(ohlcv_df)
                    except tf.MissingHistoryData as err:
                        print(f"{nowstr()} removing {base} from book due to error: {err}")
                        self.book = self.book.drop([base])
                        continue
                    tfv = ttf.tf_vectors
                    tfv.most_recent_features_only()
                    cl = cpc.performance_with_features(tfv, buy_trshld, sell_trshld)
                    # cl will be HOLD if insufficient data history is available

                    # if (base == 'ONT') and (cl == tf.TARGETS[tf.HOLD]):  # test purposes
                    #     cl = tf.TARGETS[tf.SELL]
                    if cl != tf.TARGETS[tf.BUY]:
                        # emergency sell in case no SELL signal but performance drops
                        if self.book.loc[base, 'buyprice'] > 0:
                            pricenow = ohlcv_df.loc[ohlcv_df.index[len(ohlcv_df.index)-1], "close"]
                            if self.book.loc[base, 'buyprice'] > pricenow:
                                self.book.loc[base, 'minbelow'] += 1  # minutes below buy price
                            else:
                                self.book.loc[base, 'minbelow'] = 0  # reset minute counter
                            if self.book.loc[base, 'minbelow'] > MAX_MINBELOW:  # minutes
                                print("{} Selling {} due to {} minutes < buy price of {}".format(
                                      nowstr(), sym,
                                      self.book.loc[base, 'minbelow'],
                                      self.book.loc[base, 'buyprice']))
                                cl = tf.TARGETS[tf.SELL]
                                self.book.loc[base, 'buyprice'] = 0  # reset price monitoring
                                self.book.loc[base, 'minbelow'] = 0  # minutes below buy price
                    if cl != tf.TARGETS[tf.HOLD]:
                        print(f"{nowstr()} {base} {tf.TARGET_NAMES[cl]}")
                    if cl == tf.TARGETS[tf.SELL]:
                        self.sell_order(base, ratio=1)
                    if cl == tf.TARGETS[tf.BUY]:
                        buylist.append(base) # amount to be determined by buy_ratio()

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
            self.safe_cache()
            print("finish as requested by keyboard interrupt")

    def show_all_binance_commands(self):
        binance_api = dir(ccxt.binance())
        for cmd in binance_api:
            print(cmd)

    def show_binance_base_constraints(self, base):
        dct = trd.xch.public_get_exchangeinfo()
        syms = dct["symbols"]
        for s in syms:
            if (s["baseAsset"] == base) and (s["quoteAsset"] == "USDT"):
                for f in s["filters"]:
                    print(f)


if __name__ == "__main__":
    trd = Trading()
    load_classifier = "2019-04-18_Full-SVM-orig-buy-hold-sell_gamma_0.01_extended_features_smooth+xrp_usdt5+D10"
    start_time = timeit.default_timer()
    unit_test = False
    time_aggs = {1: 10, 5: 10, 15: 10, 60: 10, 4*60: 10, 24*60: 10}
    buy_trshld = 0.8
    sell_trshld = 0.8
    cpc = cc.Cpc(5, load_classifier, None)


    # trd.buy_order('ETH', ratio=22/trd.book.loc[QUOTE, 'free'])
    # trd.sell_order('ETH', ratio=1)
    trd.trade_loop(cpc, time_aggs, buy_trshld, sell_trshld)
    trd = None  # should trigger Trading destructor

    tdiff = (timeit.default_timer() - start_time) / (60*60)
    print(f"total time: {tdiff:.2f} hours")
