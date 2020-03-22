import os
import logging
# import math
import pandas as pd
import numpy as np
import ccxt
import json
import pytz  # 3rd party: $ pip install pytz
from datetime import datetime
from env_config import Env
import env_config as env

logger = logging.getLogger(__name__)

# logger.basicConfig(level=logging.DEBUG)
RETRIES = 5  # number of ccxt retry attempts before proceeding without success


class Xch():
    """Encapsulation of ccxt and tradable cryptos and asset portfolio that is linked to
    tradable cryptos. That reduces borderplate complexity when using ccxt in this specfic
     context.

     Xch receives only actions with lowercase base using a fixed Env.quote and
     constructs symbols of `base.upper()/Env.quote.upper()` for ccxt actions.


    """
    data_keys = ["open", "high", "low", "close", "volume"]  # sequence is important, don't make it a set!
    min_daily_avg_usdt = 10000*60*24  # minimum average daily volume in USDT to be considered
    auth = None
    ohlcv = dict()
    lxch = None  # local xch
    markets = None
    quote = Env.quote.upper()

    # def __init__():
    fname = Env.auth_file
    if not os.path.isfile(Env.auth_file):
        logger.error(f"file {Env.auth_file} not found")
        # return
    try:
        with open(fname, "rb") as fp:
            auth = json.load(fp)
        if ("name" not in auth) or ("key" not in auth) or \
                ("secret" not in auth):
            logger.error(f"missing auth keys: {auth}")
    except IOError:
        logger.error(f"IO error while trying to open {fname}")
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

    def __fetch_ohlcv(sym, since, minutes):  # noqa C901
        # return Xch.lxch.fetch_ohlcv(sym, timeframe, since=since, limit=minutes)
        since = int((since.to_pydatetime() - datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds() * 1000)
        ohlcvs = None
        sym = sym.upper()
        assert minutes > 0

        for i in range(RETRIES):
            try:
                ohlcvs = Xch.lxch.fetch_ohlcv(sym, "1m", since=since, limit=minutes)
                # only 1000 entries are returned by one fetch
            except ccxt.RequestTimeout as err:
                logger.error(f"fetch_ohlcv failed {i}x due to a RequestTimeout error:", str(err))
                continue
            except ccxt.DDoSProtection as err:
                logger.error(f"fetch_ohlcv failed {i}x due to a DDoSProtection error:", str(err))
                continue
            except ccxt.ExchangeNotAvailable as err:
                logger.error(f"fetch_ohlcv failed {i}x due to a ExchangeNotAvl error:", str(err))
                continue
            except ccxt.InvalidNonce as err:
                logger.error(f"fetch_ohlcv failed {i}x due to a InvalidNonce error:", str(err))
                continue
            except ccxt.NetworkError as err:
                logger.error(f"fetch_ohlcv failed {i}x due to a NetworkError error:", str(err))
                continue
            except ccxt.ArgumentsRequired as err:
                logger.error(f"fetch_ohlcv failed {i}x due to a ArgumentsReqd error:", str(err))
                break
            except ccxt.BadRequest as err:
                logger.error(f"fetch_ohlcv failed {i}x due to a BadRequest error:", str(err))
                break
            except ccxt.NullResponse as err:
                logger.error(f"fetch_ohlcv failed {i}x due to a NullResponse error:", str(err))
                break
            except ccxt.BadResponse as err:
                logger.error(f"fetch_ohlcv failed {i}x due to a BadResponse error:", str(err))
                break
            except ccxt.AddressPending as err:
                logger.error(f"fetch_ohlcv failed {i}x due to a AddressPending error:", str(err))
                break
            except ccxt.InvalidAddress as err:
                logger.error(f"fetch_ohlcv failed {i}x due to a InvalidAddress error:", str(err))
                break
            except ccxt.NotSupported as err:
                logger.error(f"fetch_ohlcv failed {i}x due to a NotSupported error:", str(err))
                break
            except ccxt.ExchangeError as err:
                logger.error(f"fetch_ohlcv failed {i}x due to a ExchangeError error:", str(err))
                break
            else:
                if len(ohlcvs) == 0:
                    continue
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
                logger.error(f"fetch_tickers failed {i}x due to a RequestTimeout error:", str(err))
                continue
            else:
                break
        if tickers is None:
            logger.error(f"{caller} cannot fetch_tickers")
        return tickers

    def create_limit_sell_order(sym, amount, price, *params):
        # return Xch.lxch.create_limit_sell_order(sym, amount, price, *params)
        myorder = None
        for i in range(RETRIES):
            try:
                myorder = Xch.lxch.create_limit_sell_order(sym, amount, price, *params)
            except ccxt.RequestTimeout as err:
                logger.error(f"sell_order failed {i}x due to a RequestTimeout error:", str(err))
                continue
            else:
                break
        if myorder is None:
            logger.error(f"sell_order ERROR: cannot create_limit_sell_order")
        return myorder

    def create_limit_buy_order(base, amount, price, *params):
        # return Xch.lxch.create_limit_buy_order(sym, amount, price, *params)
        sym = base + "/" + Xch.quote
        myorder = None
        for i in range(RETRIES):
            try:
                myorder = Xch.lxch.create_limit_buy_order(sym, amount, price, *params)
            except ccxt.InsufficientFunds as err:
                logger.error(f"buy_order failed due to a InsufficientFunds ", str(err))
            except ccxt.RequestTimeout as err:
                logger.error(f"buy_order failed {i}x due to a RequestTimeout error: ", str(err))
                continue
            else:
                break
        if myorder is None:
            logger.error(f"buy_order ERROR: cannot create_limit_buy_order")
            return

    def myfetch_balance(caller):
        mybalance = None
        for i in range(RETRIES):
            try:
                mybalance = Xch.lxch.fetch_balance()
            except ccxt.RequestTimeout as err:
                logger.error(f"fetch_balance failed {i}x due to a RequestTimeout error:", str(err))
                continue
        if mybalance is None:
            logger.error(f"{caller} ERROR: cannot fetch_balance")
        return mybalance

    def fetch_order_book(sym):
        ob = None
        for i in range(RETRIES):
            try:
                ob = Xch.lxch.fetch_order_book(sym)  # returns 100 bids and 100 asks
            except ccxt.RequestTimeout as err:
                logger.error(f"fetch_order_book failed {i}x due to a RequestTimeout error:", str(err))
                continue
            else:
                break
        if ob is None:
            logger.error(f"log_trades_orderbook ERROR: cannot fetch_order_book")
        return ob

    def fetch_trades(sym):
        trades = None
        for i in range(RETRIES):
            try:
                trades = Xch.lxch.fetch_trades(sym)  # returns 500 trades
            except ccxt.RequestTimeout as err:
                logger.error(f"fetch_trades failed {i}x due to a RequestTimeout error:", str(err))
                continue
            else:
                break
        if trades is None:
            logger.error(f"ERROR: cannot fetch_trades")
        return trades

    def myfetch_open_orders(base, caller):
        oo = None
        sym = base + "/" + Xch.quote
        for i in range(RETRIES):
            try:
                oo = Xch.lxch.fetch_open_orders(sym)
            except ccxt.RequestTimeout as err:
                logger.error(f"fetch_open_orders failed {i}x due to a RequestTimeout error:", str(err))
                continue
            else:
                break
        if oo is None:
            logger.error(f"{caller} ERROR: cannot fetch_tickers")
        return oo

    def myfetch_cancel_orders(orderid, base):
        sym = base + "/" + Xch.quote
        for i in range(RETRIES):
            try:
                Xch.lxch.cancel_order(orderid, sym)
            except ccxt.RequestTimeout as err:
                logger.error(f"cancel_order failed {i}x due to a RequestTimeout error:", str(err))
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
                logger.error(f"2nd cancel_order failed {i}x due to a RequestTimeout error:", str(err))
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
                logger.error(f"limit violation: {sym} amount {amount} <= min({minamount})")
                return (0, price, 0)
        maxamount = Xch.markets[sym]["limits"]["amount"]["max"]
        if maxamount is not None:
            if amount >= maxamount:
                logger.error(f"limit violation: {sym} amount {amount} >= min({maxamount})")
                return (0, price, 0)
        minprice = Xch.markets[sym]["limits"]["price"]["min"]
        if minprice is not None:
            if price <= minprice:
                logger.error(f"limit violation: {sym} price {price} <= min({minprice})")
                return (amount, 0, ice_chunk)
        maxprice = Xch.markets[sym]["limits"]["price"]["max"]
        if maxprice is not None:
            if price >= maxprice:
                logger.error(f"limit violation: {sym} price {price} >= min({maxprice})")
                return (amount, 0, ice_chunk)
        cost = price * amount
        mincost = Xch.markets[sym]["limits"]["cost"]["min"]
        if mincost is not None:
            if cost <= mincost:
                logger.error(f"limit violation: {sym} cost {cost} <= min({mincost})")
                return (0, 0, 0)
        maxcost = Xch.markets[sym]["limits"]["cost"]["max"]
        if maxcost is not None:
            if cost >= maxcost:
                logger.error(f"limit violation: {sym} cost {cost} >= min({maxcost})")
                return (0, 0, 0)
        return (amount, price, ice_chunk)

    def __get_ohlcv_cache(base, when, minutes):
        start = when - pd.Timedelta(minutes, unit='T')
        df = None
        if base in Xch.ohlcv:
            df = Xch.ohlcv[base]
            df = df.drop([df.index[len(df.index)-1]])  # because the last candle is incomplete
        if df is None:
            remaining = minutes
            df = pd.DataFrame(index=pd.DatetimeIndex(pd.date_range(freq="T", start=start, periods=0, tz="UTC")),
                              dtype=np.float64, columns=Xch.data_keys)
            assert df is not None, "failed to create ohlcv df for {}-{} = {} minutes".format(
                start.strftime(Env.dt_format), when.strftime(Env.dt_format), minutes)
        else:
            df = df[Xch.data_keys]
            remaining = int((when - df.index[-1]) / pd.Timedelta(1, unit='T'))
        return df, remaining

    def __ohlcvs2df_fill_gaps(ohlcvs, df, fromdate, base):
        prev_tic = fromdate - pd.Timedelta(1, unit='T')
        count = 0
        last_tic = None
        for ohlcv in ohlcvs:
            tic = pd.Timestamp(datetime.utcfromtimestamp(ohlcv[0]/1000), tz='UTC')
            if int((tic - prev_tic)/pd.Timedelta(1, unit='T')) > 1:
                logger.info(f"ohlcv time gap for {base} between {prev_tic} and {tic}")
                if prev_tic < fromdate:
                    if df.index.isin([prev_tic]).any():  # repair first tics
                        last_tic = prev_tic
                    else:
                        # no history: then repair backwards
                        last_tic = fromdate
                        df.loc[last_tic] = ohlcv[1:6]
                        count += 1
                        prev_tic += pd.Timedelta(1, unit='T')
                prev_tic += pd.Timedelta(1, unit='T')
                while (prev_tic < tic):  # fills time gaps with lastrecord before gap
                    df.loc[prev_tic] = df.loc[last_tic]
                    count += 1
                    prev_tic += pd.Timedelta(1, unit='T')
                prev_tic -= pd.Timedelta(1, unit='T')  # correct last increment
            last_tic = tic
            df.loc[last_tic] = ohlcv[1:6]
            count += 1
            prev_tic += pd.Timedelta(1, unit='T')
        return count

    def check_df_result(df):
        last_tic = df.index[len(df)-1]
        last_tic = df.index[0]
        for ix in range(len(df)):
            tic = df.index[ix]
            diff = int(pd.Timedelta((tic - last_tic), unit='T').seconds) / 60
            if (ix > 0) and (diff != 1):
                logger.debug(f"ix: {ix} tic: {tic} - last_tic: {last_tic} != {diff} minute")
            if df.loc[tic].isna().any():
                logger.info(f"detetced Nan at ix {ix}: {df[tic]}")
            last_tic = tic
        # logger.debug("consistency check done")

    def get_ohlcv(base, minutes, last_minute):
        """Returns the last 'minutes' OHLCV values of pair before 'last_minute'.

        Is also used to calculate the average minute volume over the last hour
        that is used in buy orders to determine the max buy volume.

        Exceptions:
            1) gaps in tics (e.g. maintenance) ==> will be filled with last values before the gap,
            2) last_minute before df cache coverage (e.g. simulation) ==> # TODO
        """
        # logger.debug(f"last_minute: {last_minute}  minutes{minutes}")
        last_minute = pd.Timestamp(last_minute).replace(second=0, microsecond=0, nanosecond=0)
        last_minute += pd.Timedelta(1, unit='T')  # one minute later to include the `last_minute`
        minutes += 1  # `minutes` is extended by one to replace the last old and incomplete sample

        df, remaining = Xch.__get_ohlcv_cache(base, last_minute, minutes)
        # logger.debug(df.tail(5))
        sym = Xch.xhc_sym_of_base(base)
        while remaining > 0:
            fromdate = last_minute - pd.Timedelta(remaining, unit='T')
            # logger.error(f"{base} fromdate {env.timestr(fromdate)} minutes {remaining}")
            ohlcvs = Xch.__fetch_ohlcv(sym, since=fromdate, minutes=remaining)
            # only 1000 entries are returned by one fetch
            if ohlcvs is None:
                logger.error("{} None result when requesting {} minutes from: {}".format(
                    sym, remaining, fromdate.strftime(Env.dt_format)))
                return None

            if len(ohlcvs) > 0:
                remaining -= Xch.__ohlcvs2df_fill_gaps(ohlcvs, df, fromdate, base)
            else:
                logger.error("{} empty when requesting {} minutes from: {} - last df tic {}".format(
                    sym, remaining, fromdate.strftime(Env.dt_format), df.index[len(df)-1]))
                break

        Xch.ohlcv[base] = df
        if len(df) < minutes:
            logger.debug(f"{base} len(df) {len(df)} < {minutes} minutes")
        if len(df) > minutes:
            df = df.iloc[-minutes:]
        Xch.check_df_result(df)
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
            logger.debug(cmd)

    def __show_binance_base_constraints(base):
        dct = Xch.public_get_exchangeinfo()
        syms = dct["symbols"]
        for s in syms:
            if (s["baseAsset"] == base) and (s["quoteAsset"] == "USDT"):
                for f in s["filters"]:
                    logger.debug(f)


if __name__ == "__main__":
    # env.test_mode()
    pass
