import math
import logging
import pandas as pd
from datetime import datetime  # , timedelta  # , timezone
from env_config import nowstr
from env_config import Env
import env_config as env
import cached_crypto_data as ccd
from local_xch import Xch
import crypto_history_sets as chs

logger = logging.getLogger(__name__)

# logger.basicConfig(level=logging.DEBUG)
RETRIES = 5  # number of ccxt retry attempts before proceeding without success
BASES = ["BTC", "XRP", "ETH", "BNB", "EOS", "LTC", "NEO", "TRX", "USDT"]
# TODO can this BASES be replaced by bases from env that is equal but missing USDT?

logger.info("local_xch init")


class Bk():
    """ bookkeeping class to keep track of actions, currencies under monitoring,
        trades, and orders

        __log_action logs ccxt actions with timestamp to diagnose trading issues
        orders are logged with __log_trades_orderbook in orderbook
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
    black_bases = ["USDT", "BUSD", "TUSD"]
    actions = pd.DataFrame(columns=["ccxt", "timestamp"])  # action_id is the index nbr
    ats = None
    aid = None
    # TODO: load orderbook and trades in case they were stored
    # !orderbook and trades are only used in method __log_trades_orderbook that is not used
    orderbook = pd.DataFrame(columns=["action_id", "base", "sym", "timestamp",
                                      "side", "price", "amount"])
    trades = pd.DataFrame(columns=["action_id", "base", "sym", "timestamp",
                                   "side", "price", "amount"])
    book = None

    def __init__(self):
        Bk.exch = Xch()  # exchange.__class__
        tickers = Xch.myfetch_tickers("__init__")
        assert tickers is not None
        Bk.__update_bookkeeping(tickers)

    def __all_my_bases_with_minimum_value(mybalance, tickers):
        bases = list()
        for base in mybalance:
            sym = Xch.xhc_sym_of_base(base)
            if (base not in bases) and (base not in ["free", "info", "total", "used"]):
                if sym in tickers:
                    bval = (mybalance[base]["free"] + mybalance[base]["used"]) * tickers[sym]["last"]
                    if bval > 1:
                        bases.append(base)
                else:
                    bval = mybalance[base]["free"] + mybalance[base]["used"]
                    if bval > 0.01:  # whatever USDT value
                        #  logger.debug(f"balance value of {bval} {base} not traded in USDT")
                        pass
        return bases

    def __all_symbols_with_volume(tickers):
        bases = list()
        for sym in tickers:
            suffix = "/" + Env.quote.upper()
            if sym.endswith(suffix):  # add markets above MIN_AVG_USDT*60*24 daily volume
                base = sym[:-len(suffix)]
                if base not in Bk.black_bases:
                    if tickers[sym]["quoteVolume"] >= (Xch.min_daily_avg_usdt):
                        bases.append(base)
        return bases

    def __init_book_entries(bases, mybalance, tickers):
        xch_info = Xch.public_get_exchangeinfo()
        for base in bases:
            Bk.__update_book_entry(base, mybalance, tickers)
            Bk.book.loc[base, "buyprice"] = 0  # indicating no open buy
            Bk.book.loc[base, "minbelow"] = 0  # minutes below buy price
            syms = xch_info["symbols"]
            for s in syms:
                if (s["baseAsset"] == base) and (s["quoteAsset"] == Env.quote.upper()):
                    for f in s["filters"]:
                        if f["filterType"] == "ICEBERG_PARTS":
                            Bk.book.loc[base, "iceberg_parts"] = int(f["limit"])
                        if f["filterType"] == "LOT_SIZE":
                            Bk.book.loc[base, "lot_size_min"] = float(f["minQty"])
                            assert f["minQty"] == f["stepSize"]

    def __update_bookkeeping(tickers):
        mybalance = Bk.exch.myfetch_balance("update_bookkeeping")
        assert mybalance is not None
        Bk.__log_action("fetch_balance (update_bookkeeping)")
        bases = [base.upper() for base in Env.bases]
        bases.append(Env.quote.upper())
        Bk.book = pd.DataFrame(index=bases, columns=["action_id", "free", "used", "USDT",
                                                     "dayUSDT", "updated"])
        bases = bases + Bk.__all_my_bases_with_minimum_value(mybalance, tickers)
        bases = bases + Bk.__all_symbols_with_volume(tickers)
        Bk.__init_book_entries(bases, mybalance, tickers)
        bdf = Bk.book[["free", "used", "USDT", "dayUSDT"]]
        logger.info(str(bdf))
        for base in Bk.book.index:
            adf = None
            if base not in Bk.black_bases:
                try:  # load cached candlestick data
                    adf = ccd.load_asset_dataframe(
                        base.lower(), path=Env.cache_path, limit=chs.ActiveFeatures.history())
                except env.MissingHistoryData:
                    pass
                if adf is not None:
                    Xch.ohlcv[base] = adf

    def update_balance(tickers):
        """ Update bookkeeping of actively traded cryptos
            TODO used in sell and buy order but should not be required there
        """
        mybalance = Bk.exch.myfetch_balance("update_balance")
        if mybalance is None:
            return
        Bk.__log_action("fetch_balance (update_balance)")
        for base in Bk.book.index:
            assert base in mybalance, f"unexpectedly missing {base} in balance"
            Bk.__update_book_entry(base, mybalance, tickers)

    def __log_action(ccxt_action):
        """
        Bk.aid is an action_id to log the sequence of actions
        Bk.ats is an action_id timestamp to log the sequence of actions
        """
        Bk.ats = nowstr()  # pd.Timestamp.now(tz=Env.tz)
        Bk.aid = len(Bk.actions.index)
        Bk.actions.loc[Bk.aid] = [ccxt_action, Bk.ats]

    def __log_trades_orderbook(base):
        """
        Receives a base and a sell or buy signal.
        According to order strategy free quote will be used to fill.
        ! unused?
        """
        if base in Bk.black_bases:
            return
        sym = Xch.xhc_sym_of_base(base)
        ob = Bk.fetch_order_book(sym)
        if ob is None:
            return
        for pa in ob["bids"]:
            oix = len(Bk.orderbook.index)
            Bk.orderbook.loc[oix] = [Bk.aid, base, sym, Bk.ats, "bid", pa[0], pa[1]]
        for pa in ob["asks"]:
            oix = len(Bk.orderbook.index)
            Bk.orderbook.loc[oix] = [Bk.aid, base, sym, Bk.ats, "ask", pa[0], pa[1]]

        trades = Xch.fetch_trades(sym)
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
                df = df[ccd.data_keys]
                df = df.drop([df.index[len(df.index)-1]])  # the last candle is incomplete
                ccd.save_asset_dataframe(df, base, path=Env.cache_path)

    def __update_book_entry(base, mybalance, tickers):
        sym = Xch.xhc_sym_of_base(base)
        if (sym in tickers) or (base == Env.quote.upper()):
            Bk.book.loc[base, "action_id"] = Bk.aid
            Bk.book.loc[base, "free"] = mybalance[base]["free"]
            Bk.book.loc[base, "used"] = mybalance[base]["used"]
            if base == Env.quote.upper():
                Bk.book.loc[base, "USDT"] = 1
                Bk.book.loc[base, "dayUSDT"] = 0
            else:
                Bk.book.loc[base, "USDT"] = tickers[sym]["last"]
                Bk.book.loc[base, "dayUSDT"] = tickers[sym]["quoteVolume"]
            Bk.book.loc[base, "updated"] = pd.Timestamp.now(tz=Env.tz)

    def get_ohlcv(base, minutes, when):
        df = Bk.exch.get_ohlcv(base, minutes, when)
        Bk.__log_action("fetch_ohlcv")
        return df

    def create_limit_sell_order(sym, amount, price, *params):
        logger.info(f"sell_limit_order: {amount}{sym} at {price} {Env.quote.upper()} params:", params)
        my_order = None
        # my_order = Bk.exch.create_limit_sell_order(sym, amount, price, *params)
        Bk.__log_action("create_limit_sell_order")
        return my_order

    def create_limit_buy_order(base, amount, price, *params):
        logger.info(f"buy_limit_order: {amount}{base} at {price} {Env.quote.upper()} params:", params)
        my_order = None
        # my_order = Bk.exch.create_limit_buy_order(base, amount, price, params)
        if my_order is not None:
            if Bk.book.loc[base, "buyprice"] < price:
                Bk.book.loc[base, "buyprice"] = price
            Bk.book.loc[base, "minbelow"] = 0  # minutes below buy price
        Bk.__log_action("create_limit_buy_order")
        return my_order

    def fetch_order_book(sym):  # not yet used by Trading but foreseen for future
        ob = Bk.exch.fetch_order_book(sym)
        Bk.__log_action("fetch_order_book")
        return ob

    def fetch_trades(sym):  # not yet used by Trading but foreseen for future
        trades = Bk.exch.fetch_trades(sym)
        Bk.__log_action("fetch_trades")
        return trades

    def myfetch_open_orders(base, caller):
        oo = Bk.exch.myfetch_open_orders(base, caller)
        Bk.__log_action("fetch_open_orders")
        return oo

    def myfetch_cancel_orders(orderid, base):
        Bk.exch.myfetch_cancel_orders(orderid, base)

    def xhc_sym_of_base(base):
        return Bk.exch.xhc_sym_of_base(base)

    def myfetch_tickers(caller):
        return Bk.exch.myfetch_tickers(caller)

    def __calc_ice_chunk(base, price, amount, mincost):
        sym = base + "/" + Xch.quote
        # ice_chunk = ICEBERG_USDT_PART / price # about the chunk quantity we want
        ice_chunk = Bk.book.loc[base, "dayUSDT"] / (24 * 60 * 4)  # 1/4 of average minute volume
        ice_parts = math.ceil(amount / ice_chunk)  # about equal parts

        ice_parts = math.floor(price * amount / mincost)

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
        return ice_chunk

    def check_limits(base, amount, price):
        MINCOST = 10  # 10 USD as mincost
        ice_chunk = Bk.__calc_ice_chunk(base, price, amount, MINCOST)
        (amount, price, ice_chunk) = Bk.exch.check_limits(base, amount, price, ice_chunk)
        return (amount, price, ice_chunk)

    def trade_amount(base, ratio):
        trade_vol = 0
        for base in Bk.book.index:
            if base not in Bk.black_bases:
                trade_vol += (Bk.book.loc[base, "free"] + Bk.book.loc[base, "used"]) * \
                             Bk.book.loc[base, "USDT"]
        trade_vol = min(trade_vol, Bk.book.loc[Env.quote.upper(), "free"])
        usdt_amount = trade_vol * ratio
        return usdt_amount


if __name__ == "__main__":
    pass
