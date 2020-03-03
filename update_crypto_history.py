import pandas as pd
from env_config import Env
import env_config as env
import cached_crypto_data as ccd
# import local_xch as lxch
from local_xch import Xch
# import crypto_features as cf
import crypto_targets as ct
import condensed_features as cof
import aggregated_features as agf


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
    print("{} {} {} {}".format(env.timestr(), tf.base, tf.feature_str(), target_str))


def calc_targets_features_and_concat(stored_tf, downloaded_df, fclass):
    """ Calculates targets and features for the new downloaded ohlcv data and concatenates it with the
        already stored data, targets and features. The class 'fclass' implements the feature calculation.
    """
    if (downloaded_df is None) or (downloaded_df.index[-1] <= stored_tf.minute_data.index[-1]):
        new_minute_data = stored_tf.minute_data
    else:
        new_minute_data = pd.concat([stored_tf.minute_data, downloaded_df], sort=False)

    assert stored_tf.vec is not None
    if new_minute_data.index[-1] > stored_tf.vec.index[-1]:
        subset_df = new_minute_data.loc[
            new_minute_data.index >= (stored_tf.vec.index[-1] - pd.Timedelta(stored_tf.history(), unit='T'))]
        if "target" in subset_df:
            subset_df = subset_df.drop(columns=["target"])  # enforce recalculation of target
        subset_tf = fclass(stored_tf.base, minute_dataframe=subset_df)
        subset_tf.calc_features_and_targets()
        if stored_tf.vec.index[-1] == subset_tf.vec.index[0]:
            stored_tf.vec = stored_tf.vec.drop([stored_tf.vec.index[-1]])
        if (subset_tf.vec.index[0] - stored_tf.vec.index[-1]) == pd.Timedelta(1, unit='T'):
            new_df_vec = pd.concat([stored_tf.vec, subset_tf.vec], sort=False)
        else:
            print("unexpected gap: vec last stored: {} / first loaded: {}".format(
                stored_tf.vec.index[-1], subset_tf.vec.index[0]))
            return None, None

        # now concat minute_data including the new targets with stored data
        subset_tf.minute_data = subset_tf.minute_data.loc[subset_tf.minute_data.index >= subset_tf.vec.index[0]]
        # attention "target" of downloaded_df missing --> concat again with subset minute data including "target"
        new_minute_data = pd.concat([stored_tf.minute_data, subset_tf.minute_data], sort=False)
        ok2save = env.check_df(new_minute_data)
        if not ok2save:
            print(f"merged df.minute_data checked: {ok2save} - dataframe not saved")
            return None, None
    else:  # no new data and vec up to date
        new_df_vec = stored_tf.vec
    return new_minute_data, new_df_vec


def load_assets(bases, lastdatetime, feature_classes):
    """ Loads the cached history data as well as live data from the xch
    """
    print(bases)  # Env.usage.bases)
    for base in bases:
        print(f"supplementing {base}")

        for fclass in feature_classes:
            stored_tf = fclass(base, path=Env.data_path)
            stored_tf.calc_features_and_targets()

            downloaded_df = None
            if lastdatetime is not None:
                diffmin = int((lastdatetime - stored_tf.minute_data.index[-1])/pd.Timedelta(1, unit='T'))
                if diffmin > 0:
                    downloaded_df = Xch.get_ohlcv(base, diffmin, lastdatetime)
                    if downloaded_df is None:
                        print("skipping {}".format(base))
                        continue
                    else:
                        if stored_tf.minute_data.index[-1] == downloaded_df.index[0]:
                            # the last saved sample is incomplete and needs to be updated
                            stored_tf.minute_data = stored_tf.minute_data.drop([stored_tf.minute_data.index[-1]])
                        if (downloaded_df.index[0] - stored_tf.minute_data.index[-1]) != pd.Timedelta(1, unit='T'):
                            print("unexpected gap: minute data last stored: {} / first loaded: {}".format(
                                stored_tf.minute_data.index[-1], downloaded_df.index[0]))
                            continue

            new_minute_data, new_df_vec = calc_targets_features_and_concat(stored_tf, downloaded_df, fclass)
            if (new_minute_data is None) or (new_df_vec is None):
                print("no new data to store")
                continue
            ok2save = env.check_df(new_df_vec)
            if ok2save:
                stored_tf.vec = new_df_vec  # dirty trick: use stored_tf to save_cache of vec
                stored_tf.minute_data = new_minute_data
                check_labels(stored_tf)
                # continue  # don't save - for test purposes
                ccd.save_asset_dataframe(new_minute_data, base, path=Env.data_path)
                stored_tf.save_cache()
            else:
                print(f"merged df.vec checked: {ok2save} - dataframe not saved")


def repair_stored_ohlcv():
    df = ccd.load_asset_dataframe("xrp", Env.data_path)
    df = df.loc[df.index < pd.Timestamp("2020-01-24 10:00:00+00:00")]
    ccd.save_asset_dataframe(df, "xrp", Env.data_path)


def update_history(bases: list, last: pd.Timestamp, data_objs: list):
    """ updates saved data history up to and including 'last'
    """
    minutes = 0
    minimum = max([do.history() + 1 for do in data_objs])
    for base in bases:
        for do in data_objs:
            df = do.load_data(base)
            if not df.empty:
                minutes = int((last - df.index[0]) / pd.Timedelta(1, unit="T")) + 1
            minutes = max(minutes, minimum)
            df = do.get_data(base, last, minutes)
            do.save_data(base, df)


if __name__ == "__main__":
    # env.test_mode()
    # tee = env.Tee()
    ohlcv = ccd.Ohlcv()
    if False:  # base data repair
        df = ccd.load_asset_dataframe("xrp", Env.data_path)
        ohlcv.save_data("xrp", df)
    data_objs = [ohlcv, cof.F2cond20(ohlcv), agf.F1agg110(ohlcv), ct.T10up5low30min(ohlcv)]
    if True:
        update_history(Env.bases, pd.Timestamp.utcnow(), data_objs)
        # load_assets(Env.bases, pd.Timestamp.utcnow(), [cof.CondensedFeatures, agf.AggregatedFeatures])
        # load_assets(Env.bases, None, [cof.CondensedFeatures, agf.AggregatedFeatures])
    else:
        env.test_mode()
        base = "xrp"
        update_history([base], pd.Timestamp("2019-02-28 23:58:00+00:00"), data_objs)
    # tee.close()
