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

print("local_xch init")


class Bk():
    """ bookkeeping class to keep track of actions, currencies under monitoring,
        trades, and orders

        log_action logs ccxt actions with timestamp to diagnose trading issues
        orders are logged with log_trades_orderbook in orderbook
        book is a dataframe with bases as rows and with
        columns:
        - action_id: str describing the ccxt action
        - free: base amount that can be traded
        - used: base amount that is used in offers
        - USDT: last USDT price of base
        - dayUSDT: day quote (USDT) volume
        - updated: timestamp of update
        - buyprice: last offered buy order price (0 = no open buy offer)
        - minbelow: minutes below buy price (unused?)
        - iceberg_parts: limit of amount of iceberg parts for base
        - lot_size_min: minimum sie of single/part offer for base

    """
    actions = pd.DataFrame(columns=["ccxt", "timestamp"])  # action_id is the index nbr
    ats = None
    aid = None
    # TODO: load orderbook and trades in case they were stored
    # !orderbook and trades are only used in method log_trades_orderbook that is not used
    orderbook = pd.DataFrame(columns=["action_id", "base", "sym", "timestamp",
                                      "side", "price", "amount"])
    trades = pd.DataFrame(columns=["action_id", "base", "sym", "timestamp",
                                   "side", "price", "amount"])
    book = None

    def update_bookkeeping(tickers):
        xch_info = Xch.lxch.public_get_exchangeinfo()
        mybal = Xch.myfetch_balance("update_bookkeeping")
        if mybal is None:
            return
        Bk.log_action("fetch_balance (update_bookkeeping)")
        bases = BASES.copy()
        Bk.book = pd.DataFrame(index=bases, columns=["action_id", "free", "used", "USDT",
                                                     "dayUSDT", "updated"])
        for base in mybal:
            sym = base + "/" + Env.quote
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
            Bk.update_book_entry(base, mybal, tickers)
            Bk.book.loc[base, "buyprice"] = 0  # indicating no open buy
            Bk.book.loc[base, "minbelow"] = 0  # minutes below buy price
            syms = xch_info["symbols"]
            for s in syms:
                if (s["baseAsset"] == base) and (s["quoteAsset"] == Env.quote):
                    for f in s["filters"]:
                        if f["filterType"] == "ICEBERG_PARTS":
                            Bk.book.loc[base, "iceberg_parts"] = int(f["limit"])
                        if f["filterType"] == "LOT_SIZE":
                            Bk.book.loc[base, "lot_size_min"] = float(f["minQty"])
                            assert f["minQty"] == f["stepSize"]
        bdf = Bk.book[["free", "used", "USDT", "dayUSDT"]]
        print(bdf)
        for base in Bk.book.index:
            adf = None
            if base not in Xch.black_bases:
                try:
                    adf = ccd.load_cache_dataframe(base.lower())
                except env.MissingHistoryData:
                    pass
                if adf is not None:
                    Xch.ohlcv[base] = adf

    def update_balance(tickers):
        mybal = Xch.lxch.myfetch_balance("update_balance")
        if mybal is None:
            return
        Bk.log_action("fetch_balance (update_balance)")
        for base in Bk.book.index:
            assert base in mybal, f"unexpectedly missing {base} in balance"
            Bk.update_book_entry(base, mybal, tickers)

    def log_action(ccxt_action):
        """
        Bk.aid is an action_id to log the sequence of actions
        Bk.ats is an action_id timestamp to log the sequence of actions
        """
        Bk.ats = nowstr()  # pd.Timestamp.utcnow()
        Bk.aid = len(Bk.actions.index)
        Bk.actions.loc[Bk.aid] = [ccxt_action, Bk.ats]

    def log_trades_orderbook(base):
        """
        Receives a base and a sell or buy signal.
        According to order strategy free quote will be used to fill.
        ! unused?
        """
        if base in Xch.black_bases:
            return
        sym = base + "/" + Env.quote
        ob = None
        for i in range(RETRIES):
            try:
                ob = Xch.lxch.fetch_order_book(sym)  # returns 100 bids and 100 asks
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} fetch_order_book failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            else:
                break
        if ob is None:
            print(f"nowstr() log_trades_orderbook ERROR: cannot fetch_order_book")
            return
        Bk.log_action("fetch_order_book")
        for pa in ob["bids"]:
            oix = len(Bk.orderbook.index)
            Bk.orderbook.loc[oix] = [Bk.aid, base, sym, Bk.ats, "bid", pa[0], pa[1]]
        for pa in ob["asks"]:
            oix = len(Bk.orderbook.index)
            Bk.orderbook.loc[oix] = [Bk.aid, base, sym, Bk.ats, "ask", pa[0], pa[1]]

        trades = Xch.lxch.fetch_trades(sym)
        if trades is None:
            return
        for trade in trades:
            ix = len(Bk.trades.index)
            ts = pd.to_datetime(datetime.utcfromtimestamp(trade["timestamp"]/1000))
            Bk.trades.loc[ix] = [Bk.aid, base, sym, ts,
                                 trade["side"], trade["price"], trade["amount"]]

    def safe_cache():
        for base in Bk.book.index:
            if base in Xch.ohlcv:
                df = Xch.ohlcv[base]
                df = df[Xch.data_keys]
                df = df.drop([df.index[len(df.index)-1]])  # the last candle is incomplete
                ccd.save_asset_dataframe(df, Env.cache_path, env.sym_of_base(base))

    def update_book_entry(base, mybal, tickers):
        sym = base + "/" + Env.quote
        if (sym in tickers) or (base == Env.quote):
            Bk.book.loc[base, "action_id"] = Bk.aid
            Bk.book.loc[base, "free"] = mybal[base]["free"]
            Bk.book.loc[base, "used"] = mybal[base]["used"]
            if base == Env.quote:
                Bk.book.loc[base, "USDT"] = 1
                Bk.book.loc[base, "dayUSDT"] = 0
            else:
                Bk.book.loc[base, "USDT"] = tickers[sym]["last"]
                Bk.book.loc[base, "dayUSDT"] = tickers[sym]["quoteVolume"]
            Bk.book.loc[base, "updated"] = pd.Timestamp.utcnow()


class Xch():
    """Encapsulation of ccxt and tradable cryptos and asset portfolio that is linked to
    tradable cryptos. That reduces borderplate complexity when using ccxt in this specfic
     context.


    """
    # quote = upper(Env.quote)  # "USDT"
    black_bases = ["TUSD", "USDT", "ONG", "PAX", "BTT", "ATOM", "FET", "USDC", "ONE", "CELR", "LINK"]
    data_keys = ["open", "high", "low", "close", "volume"]
    min_daily_avg_usdt = 1500*60*24  # minimum average daily volume in USDT to be considered
    auth = None
    ohlcv = dict()
    lxch = None  # local xch
    markets = None

    # def __init__():
    fname = Env.auth_file
    if not os.path.isfile(Env.auth_file):
        print(f"file {Env.auth_file} not found")
        # return
    try:
        with open(fname, "rb") as fp:
            auth = json.load(fp)
        if ("name" not in auth) or ("key" not in auth) or \
                ("secret" not in auth):
            print(f"missing auth keys: {auth}")
    except IOError:
        print(f"IO error while trying to open {fname}")
        # return
    lxch = ccxt.binance({
        "apiKey": auth["key"],
        "secret": auth["secret"],
        "timeout": 10000,
        "enableRateLimit": True,
    })
    assert lxch.has["fetchTickers"]
    assert lxch.has["fetchOpenOrders"]
    assert lxch.has["fetchOHLCV"]
    assert lxch.has["fetchBalance"]
    assert lxch.has["fetchTickers"]
    assert lxch.has["createLimitOrder"]
    assert lxch.has["cancelOrder"]
    assert lxch.has["fetchOrderBook"]
    assert lxch.has["fetchTrades"]
    assert lxch.has["fetchMyTrades"]
    # kraken in phase 2
    # kraken = ccxt.kraken({
    #     "apiKey": "YOUR_PUBLIC_API_KEY",
    #     "secret": "YOUR_SECRET_PRIVATE_KEY",
    # })

    markets = lxch.load_markets()

    # def quote():
    #     return Xch.lxch.load_markets()

    def load_markets():
        return Xch.lxch.load_markets()

    def fetch_balance():
        return Xch.lxch.fetch_balance()

    def public_get_exchangeinfo():
        return Xch.lxch.public_get_exchangeinfo()

    def fetch_ohlcv(sym, timeframe, since, limit):
        # return Xch.lxch.fetch_ohlcv(sym, timeframe, since=since, limit=limit)
        ohlcvs = None
        sym = sym.upper()

        for i in range(RETRIES):
            try:
                ohlcvs = Xch.lxch.fetch_ohlcv(sym, "1m", since=since, limit=limit)
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
        Bk.log_action("fetch_ohlcv")
        return ohlcvs

    def fetch_open_orders(sym):
        return Xch.lxch.fetch_open_orders(sym)

    def cancel_order(orderid, sym):
        return Xch.lxch.cancel_order(orderid, sym)

    def fetch_tickers():
        return Xch.lxch.fetch_tickers()

    def myfetch_tickers(caller):
        tickers = None
        for i in range(RETRIES):
            try:
                tickers = Xch.fetch_tickers()
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} fetch_tickers failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            else:
                break
        if tickers is None:
            print(f"nowstr() {caller} ERROR: cannot fetch_tickers")
        return tickers

    def create_limit_sell_order(sym, amount, price, *params):
        # return Xch.lxch.create_limit_sell_order(sym, amount, price, *params)
        myorder = None
        for i in range(RETRIES):
            try:
                myorder = Xch.lxch.create_limit_sell_order(sym, amount, price, *params)
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} sell_order failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            else:
                break
        if myorder is None:
            print(f"nowstr() sell_order ERROR: cannot create_limit_sell_order")
        else:
            Bk.log_action("create_limit_sell_order")
        return myorder

    def create_limit_buy_order(base, amount, price, *params):
        # return Xch.lxch.create_limit_buy_order(sym, amount, price, *params)
        sym = base + "/" + Env.quote
        myorder = None
        for i in range(RETRIES):
            try:
                myorder = Xch.lxch.create_limit_buy_order(sym, amount, price, *params)
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
        if Bk.book.loc[base, "buyprice"] < price:
            Bk.book.loc[base, "buyprice"] = price
        Bk.book.loc[base, "minbelow"] = 0  # minutes below buy price
        Bk.log_action("create_limit_buy_order")

    def myfetch_balance(caller):
        mybal = None
        for i in range(RETRIES):
            try:
                mybal = Xch.lxch.fetch_balance()
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} fetch_balance failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            else:
                Bk.log_action(f"fetch_balance ({caller})")
                break
        if mybal is None:
            print(f"nowstr() {caller} ERROR: cannot fetch_balance")
        return mybal

    def fetch_trades(sym):
        trades = None
        for i in range(RETRIES):
            try:
                trades = Xch.lxch.fetch_trades(sym)  # returns 500 trades
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} fetch_trades failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            else:
                break
        if trades is None:
            print(f"nowstr() ERROR: cannot fetch_trades")
            return
        Bk.log_action("fetch_trades")

    def myfetch_open_orders(base, caller):
        oo = None
        sym = base + "/" + Env.quote
        for i in range(RETRIES):
            try:
                oo = Xch.lxch.fetch_open_orders(sym)
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} fetch_open_orders failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            else:
                break
        if oo is None:
            print(f"nowstr() {caller} ERROR: cannot fetch_tickers")
        else:
            Bk.log_action("fetch_open_orders")
        return oo

    def myfetch_cancel_orders(orderid, base):
        sym = base + "/" + Env.quote
        for i in range(RETRIES):
            try:
                Xch.lxch.cancel_order(orderid, sym)
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
                Xch.lxch.cancel_order(orderid, sym)
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} 2nd cancel_order failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
            except ccxt.OrderNotFound:
                break  # order is already closed
            else:
                break

    def check_limits(base, amount, price):
        sym = base + "/" + Env.quote
        # ice_chunk = ICEBERG_USDT_PART / price # about the chunk quantity we want
        ice_chunk = Bk.book.loc[base, "dayUSDT"] / (24 * 60 * 4)  # 1/4 of average minute volume
        ice_parts = math.ceil(amount / ice_chunk)  # about equal parts

        mincost = Xch.markets[sym]["limits"]["cost"]["min"]  # test purposes
        ice_parts = math.floor(price * amount / mincost)  # test purposes

        ice_parts = min(ice_parts, Bk.book.loc[base, "iceberg_parts"])
        if ice_parts > 1:
            ice_chunk = amount / ice_parts
            ice_chunk = int(ice_chunk / Bk.book.loc[base, "lot_size_min"]) \
                * Bk.book.loc[base, "lot_size_min"]
            ice_chunk = round(ice_chunk, Xch.markets[sym]["precision"]["amount"])
            amount = ice_chunk * ice_parts
            if ice_chunk == amount:
                ice_chunk = 0
        else:
            ice_chunk = 0
        # amount = round(amount, Xch.markets[sym]["precision"]["amount"]) is ensured by ice_chunk
        price = round(price, Xch.markets[sym]["precision"]["price"])
        minamount = Xch.markets[sym]["limits"]["amount"]["min"]
        if minamount is not None:
            if amount <= minamount:
                print(f"{nowstr()} limit violation: {sym} amount {amount} <= min({minamount})")
                return (0, price, 0)
        maxamount = Xch.markets[sym]["limits"]["amount"]["max"]
        if maxamount is not None:
            if amount >= maxamount:
                print(f"{nowstr()} limit violation: {sym} amount {amount} >= min({maxamount})")
                return (0, price, 0)
        minprice = Xch.markets[sym]["limits"]["price"]["min"]
        if minprice is not None:
            if price <= minprice:
                print(f"{nowstr()} limit violation: {sym} price {price} <= min({minprice})")
                return (amount, 0, ice_chunk)
        maxprice = Xch.markets[sym]["limits"]["price"]["max"]
        if maxprice is not None:
            if price >= maxprice:
                print(f"{nowstr()} limit violation: {sym} price {price} >= min({maxprice})")
                return (amount, 0, ice_chunk)
        cost = price * amount
        mincost = Xch.markets[sym]["limits"]["cost"]["min"]
        if mincost is not None:
            if cost <= mincost:
                print(f"{nowstr()} limit violation: {sym} cost {cost} <= min({mincost})")
                return (0, 0, 0)
        maxcost = Xch.markets[sym]["limits"]["cost"]["max"]
        if maxcost is not None:
            if cost >= maxcost:
                print(f"{nowstr()} limit violation: {sym} cost {cost} >= min({maxcost})")
                return (0, 0, 0)
        return (amount, price, ice_chunk)

    def get_ohlcv(base, minutes, when):
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
        df = None
        if base in Xch.ohlcv:
            df = Xch.ohlcv[base]
            df = df[Xch.data_keys]
            df = df.drop([df.index[len(df.index)-1]])  # because the last candle is incomplete
        sym = base + "/" + Env.quote
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
            ohlcvs = Xch.fetch_ohlcv(sym, "1m", since=since, limit=remaining)
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

        Xch.ohlcv[base] = df
        assert len(df) >= minutes, f"{base} len(df) {len(df)} < {minutes} minutes"

        if Bk.book is not None:
            isincheck = True in Bk.book.index.isin([base])
            if isincheck:
                Bk.book.loc[base, "updated"] = nowstr()
            else:
                print(f"unsupported base {base}")
        return df

    def last_hour_performance(ohlcv_df):
        cix = ohlcv_df.columns.get_loc("close")
        tix = len(ohlcv_df.index) - 1
        last_close = ohlcv_df.iat[tix, cix]
        tix = max(0, tix - 60)
        hour_close = ohlcv_df.iat[tix, cix]
        perf = (last_close - hour_close)/hour_close
        return perf

    def show_all_binance_commands():
        binance_api = dir(ccxt.binance())
        for cmd in binance_api:
            print(cmd)

    def show_binance_base_constraints(base):
        dct = Xch.lxch.public_get_exchangeinfo()
        syms = dct["symbols"]
        for s in syms:
            if (s["baseAsset"] == base) and (s["quoteAsset"] == "USDT"):
                for f in s["filters"]:
                    print(f)


def merge_asset_dataframe(path, base):
    """ extends the time range of availble usdt data into the past by mapping base_btc * btc_usdt
    """
    # "loads the object via msgpack"
    fname = path + "btc_usdt" + "_DataFrame.msg"
    btcusdt = ccd.load_asset_dataframe("btc")
    if base != "btc":
        fname = path + base + "_btc" + "_DataFrame.msg"
        basebtc = ccd.load_asset_dataframefile(fname)
        ccd.dfdescribe(f"{base}-btc", basebtc)
        fname = path + base + "_usdt" + "_DataFrame.msg"
        baseusdt = ccd.load_asset_dataframefile(fname)
        ccd.dfdescribe(f"{base}-usdt", baseusdt)
        if (baseusdt.index[0] <= basebtc.index[0]) or (baseusdt.index[0] <= btcusdt.index[0]):
            basemerged = baseusdt
        else:
            basebtc = basebtc[basebtc.index.isin(btcusdt.index)]
            basemerged = pd.DataFrame(btcusdt)
            basemerged = basemerged[basemerged.index.isin(basebtc.index)]
            for key in Xch.data_keys:
                if key != "volume":
                    basemerged[key] = basebtc[key] * btcusdt[key]
            basemerged["volume"] = basebtc.volume
            ccd.dfdescribe(f"{base}-btc-usdt", basemerged)

            baseusdt = baseusdt[baseusdt.index.isin(basemerged.index)]
            assert not baseusdt.empty
            basemerged.loc[baseusdt.index] = baseusdt[:]  # take values of cusdt where available
    else:
        basemerged = btcusdt
    ccd.dfdescribe(f"{base}-merged", basemerged)

    ccd.save_asset_dataframe(basemerged, Env.data_path, base + "usdt")

    return basemerged


def load_asset(base):
    """ Loads the cached history data as well as live data from the xch
    """
    hdf = ccd.load_asset_dataframe(base)
    ccd.dfdescribe(f"load_asset_dataframe({base})", hdf)
    ohlcv_df = Xch.get_ohlcv(base, Env.minimum_minute_df_len, datetime.utcnow())
    ccd.dfdescribe(f"Xch.get_ohlcv({base})", ohlcv_df)


if __name__ == "__main__":
    load_asset("xrp")
