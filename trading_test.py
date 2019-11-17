import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath)  # + '/../')

import timeit
import crypto_targets_features as ctf
import classify_keras as ck
import trading as trd
import numpy as np  # required to read PerfMatrix
from classify_keras import PerfMatrix, EvalPerf  # required for pickle  # noqa

def print_test():
    print("executing print_test")

def trading_test():
    print("executing trading_test")
    trd.set_environment(ctf.TestConf.test, ctf.Env.ubuntu)
    tee = ctf.Tee(f"{ck.MODEL_PATH}Log_{ctf.timestr()}.txt")
    trading = trd.Trading()
    load_classifier = "MLP-ti1-l160-h0.8-l3False-do0.8-optadam_21"
    save_classifier = None
    if True:
        cpc = ck.Cpc(load_classifier, save_classifier)
        cpc.load()
    else: # repair pickle file
        from classify_keras import PerfMatrix, EvalPerf

        print("trading pickle repair")
        cpc = ck.Cpc(load_classifier, load_classifier)
        cpc.load()
        classifier = cpc.classifier
        cpc.classifier = None  # don't save TF file again - it does not need repair
        cpc.save()
        cpc.classifier = classifier


    start_time = timeit.default_timer()
    buy_trshld = 0.7
    sell_trshld = 0.7

    # trd.buy_order("ETH", ratio=22/trd.book.loc[QUOTE, "free"])
    # trd.sell_order("ETH", ratio=1)
    # trading.trade_loop(cpc, buy_trshld, sell_trshld)
    trading = None  # should trigger Trading destructor

    tdiff = (timeit.default_timer() - start_time) / (60*60)
    print(f"total time: {tdiff:.2f} hours")
    tee.close()
