import pandas as pd
from env_config import Env
import env_config as env
import cached_crypto_data as ccd
# import local_xch as lxch
from local_xch import Xch
# import crypto_features as cf
import crypto_targets as ct
import condensed_features as cof


def OBSOLETE_merge_asset_dataframe(path, base):
    """ extends the time range of availble usdt data into the past by mapping base_btc * btc_usdt

        ! obsolete now because this was only needed to build up an extended data set when base/USDT was not
        available but base/BTC was
    """
    # "loads the object via msgpack"
    fname = path + "btc_usdt" + "_DataFrame.msg"
    btcusdt = ccd.load_asset_dataframe("btc", path=Env.data_path)
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
            for key in ccd.data_keys:
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

    ccd.save_asset_dataframe(basemerged, base, Env.data_path)

    return basemerged


def check_df(df):
    # print(df.head())
    diff = pd.Timedelta(value=1, unit="T")
    last = this = df.index[0]
    ok = True
    ix = 0
    for tix in df.index:
        if this != tix:
            print(f"ix: {ix} last: {last} tix: {tix} this: {this}")
            ok = False
            this = tix
        last = tix
        this += diff
        ix += 1
    return ok


def check_labels_of_df(df):
    # hdf = df[df.target == ct.TARGETS[ct.HOLD]]
    # hdf_pred = hdf.index[((hdf.index[1:] - hdf.index.shift(periods=1, freq="T")[1:]) != pd.Timedelta(1, unit='T'))]
    # hdf_succ = hdf.index[((hdf.index[:-1] - hdf.index.shift(periods=-1, freq="T")[:-1]) != pd.Timedelta(1, unit='T'))]

    hc = len(df[df.target == ct.TARGETS[ct.HOLD]])
    sc = len(df[df.target == ct.TARGETS[ct.SELL]])
    bc = len(df[df.target == ct.TARGETS[ct.BUY]])
    sumc = hc + sc + bc
    diff = len(df) - sumc
    target_str = f"buy={bc} sell={sc} hold={hc} sum={sumc}; total={len(df)} diff={diff}"
    unknown_df = df[(df.target != ct.TARGETS[ct.HOLD]) &
                    (df.target != ct.TARGETS[ct.SELL]) &
                    (df.target != ct.TARGETS[ct.BUY])]
    if not unknown_df.empty:
        time_udf = unknown_df.loc[
            unknown_df.index[(unknown_df.index[1:] - unknown_df.index.shift(periods=1, freq="T")[:-1]) !=
                             pd.Timedelta(1, unit='T')], "target"]
        print(time_udf.head(10))
        print("unknown_df: {}, time ranges: {}".format(len(unknown_df), len(time_udf)))
    return target_str


def check_labels(tf):
    rmd = tf.minute_data.loc[tf.minute_data.index >= tf.vec.index[0]]
    target_str = check_labels_of_df(rmd)
    print("{} {} {} {}".format(env.timestr(), tf.base, "ohlcv", target_str))
    target_str = check_labels_of_df(tf.vec)
    print("{} {} {} {}".format(env.timestr(), tf.base, tf.feature_type, target_str))


def calc_targets_features_and_concat(stored_tf, downloaded_df):
    """ Calculates targets and features for the new downloaded ohlcv data and concatenates it with the
        already stored data, targets and features.
    """
    stored_df = stored_tf.minute_data
    last = stored_df.index[len(stored_df)-1]
    if last == downloaded_df.index[0]:
        stored_df = stored_df.drop([last])  # the last saved sample is incomple and needs to be updated
        last = stored_df.index[len(stored_df)-1]
    if (downloaded_df.index[0] - last) != pd.Timedelta(1, unit='T'):
        print(f"unexpected gap: minute data last stored: {last} / first loaded: {downloaded_df.index[0]}")
        return None, None

    new_minute_data = pd.concat([stored_df, downloaded_df], sort=False)
    if stored_tf.vec is None:  # complete targets and features calculation
        if "target" in new_minute_data:
            new_minute_data = new_minute_data.drop(columns=["target"])  # enforce recalculation of target
        subset_tf = cof.CondensedFeatures(stored_tf.base, minute_dataframe=new_minute_data)
        subset_tf.calc_features_and_targets()
        new_df_vec = subset_tf.vec
    else:  # only targets and features calculation for the downloaded part
        subset_df = new_minute_data.loc[
            new_minute_data.index >= (downloaded_df.index[0] - pd.Timedelta(cof.MHE, unit='T'))]
        if "target" in subset_df:
            subset_df = subset_df.drop(columns=["target"])  # enforce recalculation of target
        subset_tf = cof.CondensedFeatures(stored_tf.base, minute_dataframe=subset_df)
        subset_tf.calc_features_and_targets()
        last = stored_tf.vec.index[len(stored_tf.vec)-1]
        if last == subset_tf.vec.index[0]:
            stored_tf.vec = stored_tf.vec.drop([last])
            last = stored_tf.vec.index[len(stored_tf.vec)-1]
        if (subset_tf.vec.index[0] - last) == pd.Timedelta(1, unit='T'):
            new_df_vec = pd.concat([stored_tf.vec, subset_tf.vec], sort=False)
        else:
            print("unexpected gap: vec last stored: {} / first loaded: {}".format(last, subset_tf.vec.index[0]))
            return None, None

        # now concat minute_data including the new targets with stored data
        subset_tf.minute_data = subset_tf.minute_data.loc[subset_tf.minute_data.index >= subset_tf.vec.index[0]]
        new_minute_data = pd.concat([stored_df, subset_tf.minute_data], sort=False)
        ok2save = check_df(new_minute_data)  # attention "target" of downloaded_df missing
        if not ok2save:
            print(f"merged df.minute_data checked: {ok2save} - dataframe not saved")
            return None, None
    return new_minute_data, new_df_vec


def load_asset(bases):
    """ Loads the cached history data as well as live data from the xch
    """
    print(bases)  # Env.usage.bases)
    for base in bases:
        print(f"supplementing {base}")

        stored_tf = cof.CondensedFeatures(base, path=Env.data_path)
        stored_tf.calc_features_and_targets()
        stored_df = stored_tf.minute_data
        # stored_df = ccd.load_asset_dataframe(base, path=Env.data_path)
        # stored_df.index.tz_localize(tz='UTC')

        last = stored_df.index[len(stored_df)-1]
        now = pd.Timestamp.utcnow()
        diffmin = int((now - last)/pd.Timedelta(1, unit='T'))
        downloaded_df = Xch.get_ohlcv(base, diffmin, now)
        if downloaded_df is None:
            print("skipping {}".format(base))
            continue

        new_minute_data, new_df_vec = calc_targets_features_and_concat(stored_tf, downloaded_df)
        if (new_minute_data is None) or (new_df_vec is None):
            print("processing downloaded data failed")
            continue
        ok2save = check_df(new_df_vec)
        if ok2save:
            stored_tf.vec = new_df_vec  # dirty trick: use stored_tf to save_cache of vec
            stored_tf.minute_data = new_minute_data
            check_labels(stored_tf)
            # continue  # don't save - for test purposes
            ccd.save_asset_dataframe(new_minute_data, base, path=Env.data_path)
            stored_tf.save_cache()
        else:
            print(f"merged df.vec checked: {ok2save} - dataframe not saved")


if __name__ == "__main__":
    # env.test_mode()
    if True:
        load_asset(Env.usage.bases)
    else:
        base = "xrp"
        load_asset([base])

        # tf = cof.CondensedFeatures(base, path=Env.data_path)
        # tf.calc_features_and_targets()
        # tf.minute_data = tf.minute_data.loc[tf.minute_data.index < pd.Timestamp("2020-01-14 21:54:00+00:00")]
        # tf.vec = tf.vec.loc[tf.vec.index < pd.Timestamp("2020-01-14 21:54:00+00:00")]
        # ccd.save_asset_dataframe(tf.minute_data, base, path=Env.data_path)
        # tf.save_cache()
