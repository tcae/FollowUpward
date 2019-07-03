        def add_targets_stripped(time_agg, df):
            "target = achieved if improvement > 1% without intermediate loss of more than 0.2%"

            def close_delta_ratio(tix1, tix2, cix):
                assert tix1 <= tix2
                last_close = df.iat[tix1, cix]
                this_close = df.iat[tix2, cix]
                delta = (this_close - last_close) / last_close
                return delta

            #print(f"{datetime.now()}: add_targets {time_agg}")
            df["target"] = TARGETS[HOLD]
            lix = df.columns.get_loc("target")
            cix = df.columns.get_loc("close")
            win = dict()
            loss = dict()
            lossix = dict()
            winix = dict()
            for slot in range(0, time_agg):
                win[slot] = loss[slot] = 0.
                winix[slot] = lossix[slot] = slot
            for tix in range(time_agg, len(df), 1):  # tix = time index
                slot = (tix % time_agg)
                delta = close_delta_ratio(tix - time_agg, tix, cix)
                if delta < 0:
                    if loss[slot] < 0:  # loss monitoring is running
                        loss[slot] = close_delta_ratio(lossix[slot], tix, cix) # DIFF does not accumulate inaccurate deltas but calcs new 
                    else:  # first time bar of decrease period
                        lossix[slot] = tix
                        loss[slot] = delta
                    if win[slot] > 0:  # win monitoring is running
                        win[slot] = close_delta_ratio(winix[slot], tix, cix) # DIFF does not accumulate inaccurate deltas but calcs new
                        if win[slot] < 0:  # reset win monitor because it is below start price
                            win[slot] = 0.
                    while loss[slot] < SELL_THRESHOLD:
                        # address scenario of multiple small decreases and then a big decrease step -> multiple SELL signals required
                        win[slot] = 0.  # reset win monitor -> dip exceeded threshold
                        df.iat[lossix[slot], lix] = TARGETS[SELL]
                        if (lossix[slot] + time_agg) > tix:
                            break
                        lossix[slot] += time_agg
                        loss[slot] = close_delta_ratio(lossix[slot], tix, cix) # DIFF does not accumulate inaccurate deltas but calcs new
                elif delta > 0:
                    if win[slot] > 0:  # win monitoring is running
                        win[slot] = close_delta_ratio(winix[slot], tix, cix) # DIFF does not accumulate inaccurate deltas but calcs new
                    else:  # first time bar of increase period
                        winix[slot] = tix
                        win[slot] = delta
                    if loss[slot] < 0:  # loss monitoring is running
                        loss[slot] = close_delta_ratio(lossix[slot], tix, cix) # DIFF does not accumulate inaccurate deltas but calcs new
                        if loss[slot] > 0:
                            loss[slot] = 0.  # reset loss monitor -> recovered before sell threshold
                    while win[slot] > BUY_THRESHOLD:
                        loss[slot] = 0.  # reset win monitor -> dip exceeded threshold
                        df.iat[winix[slot], lix] = TARGETS[BUY]
                        if (winix[slot] + time_agg) > tix:
                            break
                        winix[slot] += time_agg
                        win[slot] = close_delta_ratio(winix[slot], tix, cix) # DIFF does not accumulate inaccurate deltas but calcs new
            # report_setsize("complete set", df)
