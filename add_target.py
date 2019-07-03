        def add_targets(time_agg, df):
            "target = achieved if improvement > 1% without intermediate loss of more than 0.2%"
            #print(f"{datetime.now()}: add_targets {time_agg}")
            df['target'] = TARGETS[HOLD]
            lix = df.columns.get_loc('target')
            cix = df.columns.get_loc('close')
            win = dict()
            loss = dict()
            lossix = dict()
            winix = dict()
            ixfifo = Queue()  # will hold all sell ix to smooth out if dip sell does no tpay off
            closeatsell = closeatbuy = 0
            lasttarget = dict()
            for slot in range(0, time_agg):
                win[slot] = loss[slot] = 0.
                winix[slot] = lossix[slot] = slot
                lasttarget[slot] = TARGETS[HOLD]
            for tix in range(time_agg, len(df), 1):  # tix = time index
                slot = (tix % time_agg)
                last_close = df.iat[tix - time_agg, cix]
                this_close = df.iat[tix, cix]
                delta = (this_close - last_close) / last_close  # * 1000 no longer in per mille
                if delta < 0:
                    if loss[slot] < 0:  # loss monitoring is running
                        loss[slot] += delta
                    else:  # first time bar of decrease period
                        lossix[slot] = tix
                        loss[slot] = delta
                    if win[slot] > 0:  # win monitoring is running
                        win[slot] += delta
                        if win[slot] < 0:  # reset win monitor because it is below start price
                            win[slot] = 0.
                    if loss[slot] < SELL_THRESHOLD:  # reset win monitor -> dip exceeded threshold
                        win[slot] = 0.
                        df.iat[lossix[slot], lix] = lasttarget[slot] = TARGETS[SELL]
                        lossix[slot] += 1  # allow multiple signals if conditions hold => FIX this changes slot!
                        # FIX loss[slot] is not corrected with the index change
                        #  here comes the smooth execution for BUY peaks:
                        if closeatbuy > 0:  # smoothing is active
                            buy_sell = -2 * (FEE + TRADE_SLIP) + this_close - closeatbuy
                            while not ixfifo.empty():
                                smooth_ix = ixfifo.get()
                                if buy_sell < 0:
                                    # if fee loss more than dip loss/gain then smoothing
                                    df.iat[smooth_ix, lix] = TARGETS[HOLD]
                            closeatbuy = 0
                        #  here comes the smooth preparation for SELL dips:
                        if closeatsell == 0:
                            closeatsell = this_close
                        ixfifo.put(tix)  # prep after execution due to queue reuse
                elif delta > 0:
                    if win[slot] > 0:  # win monitoring is running
                        win[slot] += delta
                    else:  # first time bar of increase period
                        winix[slot] = tix
                        win[slot] = delta
                    if loss[slot] < 0:  # loss monitoring is running
                        loss[slot] += delta
                        if loss[slot] > 0:
                            loss[slot] = 0.  # reset loss monitor -> recovered before sell threshold
                    if win[slot] > BUY_THRESHOLD:  # reset win monitor -> dip exceeded threshold
                        loss[slot] = 0.
                        df.iat[winix[slot], lix] = lasttarget[slot] = TARGETS[BUY]
                        winix[slot] += 1  # allow multiple signals if conditions hold => FIX this changes slot!
                        # FIX win[slot] is not corrected with the index change
                        #  here comes the smooth execution for SELL dips:
                        if closeatsell > 0:  # smoothing is active
                            sell_buy = -2 * (FEE + TRADE_SLIP)
                            holdgain = this_close - closeatsell
                            while not ixfifo.empty():
                                smooth_ix = ixfifo.get()
                                if sell_buy < holdgain:
                                    # if fee loss more than dip loss/gain then smoothing
                                    df.iat[smooth_ix, lix] = TARGETS[HOLD]
                            closeatsell = 0
                        #  here comes the smooth preparation for BUY peaks:
                        if closeatbuy == 0:
                            closeatbuy = this_close
                        ixfifo.put(tix)  # prep after execution due to queue reuse
            # report_setsize("complete set", df)

