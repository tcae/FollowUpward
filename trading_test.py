import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath)  # + '/../')

import timeit
import crypto_targets_features as ctf
import classify_keras as ck
import trading as trd
from classify_keras import PerfMatrix, EvalPerf  # required for pickle  # noqa


def trading_test():
    trd.set_environment(ctf.TestConf.test, ctf.Env.ubuntu)
    tee = ctf.Tee(f"{ck.MODEL_PATH}Log_{ctf.timestr()}.txt")
    trading = trd.Trading()
    load_classifier = "MLP-ti1-l160-h0.8-l3False-do0.8-optadam_21"
    save_classifier = None
    cpc = ck.Cpc(load_classifier, save_classifier)
    cpc.load()

    start_time = timeit.default_timer()
    buy_trshld = 0.7
    sell_trshld = 0.7

    # trd.buy_order("ETH", ratio=22/trd.book.loc[QUOTE, "free"])
    # trd.sell_order("ETH", ratio=1)
    trading.trade_loop(cpc, buy_trshld, sell_trshld)
    trading = None  # should trigger Trading destructor

    tdiff = (timeit.default_timer() - start_time) / (60*60)
    print(f"total time: {tdiff:.2f} hours")
    tee.close()


if __name__ == "__main__":
    trading_test()
