import os
# import math
import pandas as pd
import numpy as np
import ccxt
import json
import pytz  # 3rd party: $ pip install pytz
from datetime import datetime, timedelta  # , timezone
from env_config import nowstr
from env_config import Env
import env_config as env
import cached_crypto_data as ccd

# logging.basicConfig(level=logging.DEBUG)
RETRIES = 5  # number of ccxt retry attempts before proceeding without success


class Xch():
    """Encapsulation of ccxt and tradable cryptos and asset portfolio that is linked to
    tradable cryptos. That reduces borderplate complexity when using ccxt in this specfic
     context.

     Xch receives only actions with lowercase base using a fixed Env.quote and
     constructs symbols of `base.upper()/Env.quote.upper()` for ccxt actions.


    """
    data_keys = ["open", "high", "low", "close", "volume"]
    min_daily_avg_usdt = 1500*60*24  # minimum average daily volume in USDT to be considered
    auth = None
    ohlcv = dict()
    lxch = None  # local xch
    markets = None
    quote = Env.quote.upper()

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
        "rateLimit": 250,
        "enableRateLimit": False
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

    def xhc_sym_of_base(base):
        sym = env.sym_of_base(base)
        sym = sym.replace(Env.sym_sep, Env.xch_sym_sep)
        sym = sym.upper()
        return sym

    def load_markets():
        return Xch.lxch.load_markets()

    def fetch_balance():
        return Xch.lxch.fetch_balance()

    def public_get_exchangeinfo():
        return Xch.lxch.public_get_exchangeinfo()

    def __fetch_ohlcv(sym, timeframe, since, limit):  # noqa C901
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
        return myorder

    def create_limit_buy_order(base, amount, price, *params):
        # return Xch.lxch.create_limit_buy_order(sym, amount, price, *params)
        sym = base + "/" + Xch.quote
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

    def myfetch_balance(caller):
        mybalance = None
        for i in range(RETRIES):
            try:
                mybalance = Xch.lxch.fetch_balance()
            except ccxt.RequestTimeout as err:
                print(f"{nowstr()} fetch_balance failed {i}x due to a RequestTimeout error:",
                      str(err))
                continue
        if mybalance is None:
            print(f"nowstr() {caller} ERROR: cannot fetch_balance")
        return mybalance

    def fetch_order_book(sym):
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
        return ob

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
        return trades

    def myfetch_open_orders(base, caller):
        oo = None
        sym = base + "/" + Xch.quote
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
        return oo

    def myfetch_cancel_orders(orderid, base):
        sym = base + "/" + Xch.quote
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

    def check_limits(base, amount, price, ice_chunk):  # noqa C901  all checks in an overview is OK
        sym = base + "/" + Xch.quote
        mincost = Xch.markets[sym]["limits"]["cost"]["min"]
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

    def __get_ohlcv_cache(base, when, minutes):
        start = when - timedelta(minutes=minutes-1)
        df = None
        if base in Xch.ohlcv:
            df = Xch.ohlcv[base]
            df = df[Xch.data_keys]
            df = df.drop([df.index[len(df.index)-1]])  # because the last candle is incomplete
        remaining = minutes
        if df is None:
            # df = pd.DataFrame(index=pd.DatetimeIndex(freq="T", start=start, end=when, tz="UTC"),
            df = pd.DataFrame(index=pd.DatetimeIndex(pd.date_range(freq="T", start=start, end=when, tz="UTC")),
                              dtype=np.float64, columns=Xch.data_keys)
            assert df is not None, "failed to create ohlcv df for {}-{} = {} minutes".format(
                start.strftime(Env.dt_format), when.strftime(Env.dt_format), minutes)
        else:
            last_tic = df.index[len(df.index)-1].to_pydatetime()
            dtlast = last_tic  # ! .replace(tzinfo=None)
            dfdiff = int((when - dtlast) / timedelta(minutes=1))
            if dfdiff < remaining:
                remaining = dfdiff
        return df, remaining

    def get_ohlcv(base, minutes, when):
        """Returns the last 'minutes' OHLCV values of pair before 'when'.

        Is also used to calculate the average minute volume over the last hour
        that is used in buy orders to determine the max buy volume.

        Exceptions:
            1) gaps in tics (e.g. maintenance) ==> will be filled with last values before the gap,
            2) when before df cache coverage (e.g. simulation) ==> # TODO
        """
        when = pd.Timestamp(when).replace(second=0, microsecond=0, nanosecond=0)  # ! , tzinfo=None)
        when = when.to_pydatetime()
        when = when.replace(tzinfo=pytz.utc)
        minutes += 1
        df, remaining = Xch.__get_ohlcv_cache(base, when, minutes)
        sym = Xch.xhc_sym_of_base(base)
        count = 0
        while remaining > 0:
            fromdate = when - timedelta(minutes=remaining-1)
            since = int((fromdate - datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds() * 1000)
            # print(f"{nowstr()} {base} fromdate {env.timestr(fromdate)} minutes {remaining}")
            ohlcvs = Xch.__fetch_ohlcv(sym, "1m", since=since, limit=remaining)
            # only 1000 entries are returned by one fetch
            if ohlcvs is None:
                print(
                    "{} get_ohlcv ERROR: {} None result when requesting {} minutes from: {}".format(
                        nowstr(), sym, remaining, fromdate.strftime(Env.dt_format)))
                return None
            if len(ohlcvs) == 0:
                print(
                    "{} get_ohlcv ERROR: {} empty when requesting {} minutes from: {}".format(
                        nowstr(), sym, remaining, fromdate.strftime(Env.dt_format)))
                return None
            prev_tic = fromdate - timedelta(minutes=1)
            itic = None
            lastohlcv = None
            for ohlcv in ohlcvs:
                tic = pd.to_datetime(datetime.utcfromtimestamp(ohlcv[0]/1000))
                tic = tic.replace(tzinfo=pytz.utc)
                if int((tic - prev_tic)/timedelta(minutes=1)) > 1:
                    print(f"ohlcv time gap for {base} between {prev_tic} and {tic}")
                    if prev_tic < fromdate:  # repair first tics
                        print(f"no repair of missing history data for {base}")
                        return None
                        prev_tic += timedelta(minutes=1)
                        iptic = itic = pd.Timestamp(prev_tic).tz_convert(tz="UTC")
                        df.loc[iptic] = [ohlcv[1], ohlcv[2], ohlcv[3], ohlcv[4], ohlcv[5]]
                        df.loc[iptic, "volume"] = 0  # indicate gap fillers with volume == 0
                        remaining -= 1
                    prev_tic += timedelta(minutes=1)
                    while (prev_tic < tic):
                        iptic = pd.Timestamp(prev_tic).tz_convert(tz="UTC")
                        df.loc[iptic] = df.loc[itic]
                        remaining -= 1
                        prev_tic += timedelta(minutes=1)
                itic = pd.Timestamp(tic).tz_convert("UTC")
                df.loc[itic] = [ohlcv[1], ohlcv[2], ohlcv[3], ohlcv[4], ohlcv[5]]
                remaining -= 1
                prev_tic += timedelta(minutes=1)
                count += 1
                if lastohlcv == ohlcv:
                    print(f"no progress count: {count}  ohlv: {ohlcv}  lastohlcv: {lastohlcv}")
                lastohlcv = ohlcv

        Xch.ohlcv[base] = df
        if len(df) < minutes:
            print(f"{base} len(df) {len(df)} < {minutes} minutes")
        return df

    def __last_hour_performance(ohlcv_df):
        cix = ohlcv_df.columns.get_loc("close")
        tix = len(ohlcv_df.index) - 1
        last_close = ohlcv_df.iat[tix, cix]
        tix = max(0, tix - 60)
        hour_close = ohlcv_df.iat[tix, cix]
        perf = (last_close - hour_close)/hour_close
        return perf

    def __show_all_binance_commands():
        binance_api = dir(ccxt.binance())
        for cmd in binance_api:
            print(cmd)

    def __show_binance_base_constraints(base):
        dct = Xch.public_get_exchangeinfo()
        syms = dct["symbols"]
        for s in syms:
            if (s["baseAsset"] == base) and (s["quoteAsset"] == "USDT"):
                for f in s["filters"]:
                    print(f)


def OBSOLETE_merge_asset_dataframe(path, base):
    """ extends the time range of availble usdt data into the past by mapping base_btc * btc_usdt

        ! obsolete now because this was only needed to build up an extended data set when base/USDT was not
        available but base/BTC was
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

            baseusdt = baseusdt[baseusdt.index.isin(basemerged.index)]  # ! why limit to basebtc range?
            assert not baseusdt.empty
            basemerged.loc[baseusdt.index] = baseusdt[:]  # take values of cusdt where available
    else:
        basemerged = btcusdt
    ccd.dfdescribe(f"{base}-merged", basemerged)

    ccd.save_asset_dataframe(basemerged, Env.data_path, base + "usdt")

    return basemerged


def check_df(df):
    diff = pd.Timedelta(value=1, unit="m")
    last = this = df.index[0]
    ok = True
    for tix in df.index:
        if this != tix:
            print(f"last: {last} tix: {tix} this: {this}")
            ok = False
            this = tix
        last = tix
        this += diff
    return ok


def load_asset(base):
    """ Loads the cached history data as well as live data from the xch
    """

    # now = datetime.utcnow()
    # diffmin = 2000
    # ohlcv_df = Xch.get_ohlcv(base, diffmin, now)
    # ccd.dfdescribe(f"ohlcv_df({diffmin})", ohlcv_df)
    # print(f"loaded asset df checked: {check_df(ohlcv_df)}")

    # ohlcv_df = ohlcv_df.drop([ohlcv_df.index[diffmin-2]])
    # ccd.dfdescribe(f"ohlcv_df({diffmin-1})", ohlcv_df)
    # print(f"dropped asset df checked: {check_df(ohlcv_df)}")
    # return

    print(Env.usage.bases)  # Env.usage.bases)
    for base in Env.usage.bases:
        print(f"supplementing {base}")
        hdf = ccd.load_asset_dataframe(base)

        last = (hdf.index[len(hdf)-1]).to_pydatetime()
        last = last.replace(tzinfo=None)
        now = datetime.utcnow()
        diffmin = int((now - last)/timedelta(minutes=1))
        ohlcv_df = Xch.get_ohlcv(base, diffmin, now)
        if ohlcv_df is None:
            print("skipping {}".format(base))
            continue
        tix = hdf.index[len(hdf)-1]
        while tix == ohlcv_df.index[0]:
            ohlcv_df = ohlcv_df.drop([tix])
        hdf = pd.concat([hdf, ohlcv_df], sort=False)
        ok2save = check_df(hdf)
        print(f"merged df checked: {ok2save}")
        if ok2save:
            ccd.save_asset_dataframe(hdf, base)


if __name__ == "__main__":
    load_asset("xrp")
