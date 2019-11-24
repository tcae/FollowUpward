import os
import math
import pandas as pd
import ccxt
import json
from datetime import datetime, timedelta  # , timezone
from env_config import nowstr
from env_config import Env
import env_config as env
import cached_crypto_data as ccd

# logging.basicConfig(level=logging.DEBUG)
RETRIES = 5  # number of ccxt retry attempts before proceeding without success
BASES = ["BTC", "XRP", "ETH", "BNB", "EOS", "LTC", "NEO", "TRX", "USDT"]
# TODO can this BASES be replaced by bases from env that is equal but missing USDT?


class Xch():
    """Encapsulation of ccxt and tradable cryptos and asset portfolio that is linked to
    tradable cryptos. That reduces borderplate complexity when using ccxt in this specfic
     context.


    """
    quote = "USDT"
    black_bases = ["TUSD", "USDT", "ONG", "PAX", "BTT", "ATOM", "FET", "USDC", "ONE", "CELR", "LINK"]
    data_keys = ["open", "high", "low", "close", "volume"]
    min_daily_avg_usdt = 1500*60*24  # minimum average daily volume in USDT to be considered

    def __init__(self):
        fname = Env.auth_file
        if not os.path.isfile(Env.auth_file):
            print(f"file {Env.auth_file} not found")
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
        self.ohlcv = dict()
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
        # TODO: load orderbook and trades in case they were stored
        self.orderbook = pd.DataFrame(columns=["action_id", "base", "sym", "timestamp",
                                               "side", "price", "amount"])
        self.trades = pd.DataFrame(columns=["action_id", "base", "sym", "timestamp",
                                            "side", "price", "amount"])

    # def quote(self):
    #     return self.xch.load_markets()

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
            sym = base + "/" + Xch.quote
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
                if base not in Xch.black_bases:
                    if tickers[sym]["quoteVolume"] >= (Xch.min_daily_avg_usdt):
                        bases.append(base)
        for base in bases:
            self.update_book_entry(base, mybal, tickers)
            self.book.loc[base, "buyprice"] = 0  # indicating no open buy
            self.book.loc[base, "minbelow"] = 0  # minutes below buy price
            syms = xch_info["symbols"]
            for s in syms:
                if (s["baseAsset"] == base) and (s["quoteAsset"] == Xch.quote):
                    for f in s["filters"]:
                        if f["filterType"] == "ICEBERG_PARTS":
                            self.book.loc[base, "iceberg_parts"] = int(f["limit"])
                        if f["filterType"] == "LOT_SIZE":
                            self.book.loc[base, "lot_size_min"] = float(f["minQty"])
                            assert f["minQty"] == f["stepSize"]
        bdf = self.book[["free", "used", "USDT", "dayUSDT"]]
        print(bdf)
        for base in self.book.index:
            adf = None
            if base not in Xch.black_bases:
                try:
                    adf = ccd.load_cache_dataframe(base.lower())
                except env.MissingHistoryData:
                    pass
                if adf is not None:
                    self.ohlcv[base] = adf

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

    def myfetch_tickers(self, caller):
        tickers = None
        for i in range(RETRIES):
            try:
                tickers = self.fetch_tickers()
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} fetch_tickers failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            else:
                break
        if tickers is None:
            print(f"nowstr() {caller} ERROR: cannot fetch_tickers")
        return tickers

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
        sym = base + "/" + Xch.quote
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
        if base in Xch.black_bases:
            return
        sym = base + "/" + Xch.quote
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

    def safe_cache(self):
        for base in self.book.index:
            if base in self.ohlcv:
                df = self.ohlcv[base]
                df = df[Xch.data_keys]
                df = df.drop([df.index[len(df.index)-1]])  # the last candle is incomplete
                ccd.save_asset_dataframe(df, Env.cache_path, env.sym_of_base(base))

    def update_book_entry(self, base, mybal, tickers):
        sym = base + "/" + Xch.quote
        if (sym in tickers) or (base == Xch.quote):
            self.book.loc[base, "action_id"] = self.aid
            self.book.loc[base, "free"] = mybal[base]["free"]
            self.book.loc[base, "used"] = mybal[base]["used"]
            if base == Xch.quote:
                self.book.loc[base, "USDT"] = 1
                self.book.loc[base, "dayUSDT"] = 0
            else:
                self.book.loc[base, "USDT"] = tickers[sym]["last"]
                self.book.loc[base, "dayUSDT"] = tickers[sym]["quoteVolume"]
            self.book.loc[base, "updated"] = pd.Timestamp.utcnow()

    def myfetch_open_orders(self, base, caller):
        oo = None
        sym = base + "/" + Xch.quote
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
        sym = base + "/" + Xch.quote
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
        sym = base + "/" + Xch.quote
        # ice_chunk = ICEBERG_USDT_PART / price # about the chunk quantity we want
        ice_chunk = self.book.loc[base, "dayUSDT"] / (24 * 60 * 4)  # 1/4 of average minute volume
        ice_parts = math.ceil(amount / ice_chunk)  # about equal parts

        mincost = self.markets[sym]["limits"]["cost"]["min"]  # test purposes
        ice_parts = math.floor(price * amount / mincost)  # test purposes

        ice_parts = min(ice_parts, self.book.loc[base, "iceberg_parts"])
        if ice_parts > 1:
            ice_chunk = amount / ice_parts
            ice_chunk = int(ice_chunk / self.book.loc[base, "lot_size_min"]) \
                * self.book.loc[base, "lot_size_min"]
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

    def get_ohlcv(self, base, minutes, when):
        """Returns the last 'minutes' OHLCV values of pair before 'when'.

        Is also used to calculate the average minute volume over the last hour
        that is used in buy orders to determine the max buy volume.

        Exceptions:
            1) gaps in tics (e.g. maintenance) ==> will be filled with last values before the gap,
            2) when before df cache coverage (e.g. simulation) ==> # TODO
        """
        when = pd.Timestamp(when).replace(second=0, microsecond=0, nanosecond=0, tzinfo=None)
        when = when.to_pydatetime()
        minutes += 1
        isincheck = True in self.book.index.isin([base])
        df = None
        if isincheck:
            if base in self.ohlcv:
                df = self.ohlcv[base]
                df = df[Xch.data_keys]
                df = df.drop([df.index[len(df.index)-1]])  # because the last candle is incomplete
            sym = base + "/" + Xch.quote
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
                # print(f"{nowstr()} {base} fromdate {env.timestr(fromdate)} minutes {remaining}")
                ohlcvs = self.fetch_ohlcv(sym, "1m", since=since, limit=remaining)
                # only 1000 entries are returned by one fetch
                if ohlcvs is None:
                    return None
                prev_tic = fromdate - timedelta(minutes=1)
                itic = None
                for ohlcv in ohlcvs:
                    if df is None:
                        df = pd.DataFrame(columns=Xch.data_keys)
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
            self.book.loc[base, "updated"] = nowstr()
        else:
            print(f"unsupported base {base}")
        return df

    def last_hour_performance(self, ohlcv_df):
        cix = ohlcv_df.columns.get_loc("close")
        tix = len(ohlcv_df.index) - 1
        last_close = ohlcv_df.iat[tix, cix]
        tix = max(0, tix - 60)
        hour_close = ohlcv_df.iat[tix, cix]
        perf = (last_close - hour_close)/hour_close
        return perf

    def show_all_binance_commands(self):
        binance_api = dir(ccxt.binance())
        for cmd in binance_api:
            print(cmd)

    def show_binance_base_constraints(self, base):
        dct = self.myxch.public_get_exchangeinfo()
        syms = dct["symbols"]
        for s in syms:
            if (s["baseAsset"] == base) and (s["quoteAsset"] == "USDT"):
                for f in s["filters"]:
                    print(f)
