# FollowUpward

Follow crypto trend - a python and ML learning project

## Vision

A development environment that supports engineering features and classifier nets crypto trading decisions.
Stock trading is a possible future step but cryoto trading offers more opportunities (and risks) due to higher volatility.

## Environment

- Tensorflow 2 servings Keras for machine learning.
- ccxt for crypto trading.

## Main code entry points

- `classify_keras.py main` is used to train and evaluate ML.
- `trading.py main` is used to trade via ccxt.
- `training_dashboard and live_dashboard` are for visual analysis
- `local_xch` updates the historic prices

//: # - `Catalyst2PandasData.ipynb` to convert historical catalyst data to pandas files in msgpack format (due to pickle incompatibilties with pandas)

## get historical crypto data

`history2pandas.py` to load historical Catalyst data into pandas stored as msgpack files.
//: # `Catalyst2PandasData.ipynb` to load historical Catalyst data into pandas stored as msgpack files.

## machine learning

`classify_keras.py` to train and evaluate neural nets trained via the keras interface by TensorFlow with main classes:

- Cpc to represent a crupto perforamce classifier
- PerfMatrix to

`crypto_targets_features.py` to provide dample features and their target label and to manage sample sets with the main classes:

- TargetsFeatures (crypto_targets_features.py) to implement classifier features and their target class labels.
- HistorySets (crypto_targets_features.py) to split sets in training, evaluation and test sets. These are time blocks of some weeks because a sample wise scrabling would leak too much labeling due to the samples history features use.

## trading

`trading.py` is the central class for crypto trading via `ccxt`. It contains and manges a state machine and uses the trained neural net to classify the trading signal based on the stream of incoming market data.
`trading` implements the trading strategy and abstracts from handling details. It uses `Bk` for bookkeeping (trade book, order book, mirror of assets, logging of actions).
`Bk` uses in turn `Xch` that encapsulates the handling details with `ccxt` such as timeout handling, exception handling, limit checking, converting symbols to lowercase
while trading only uses base in lowercase and always trades aganist a single `Env.quote`.

### configurations

- black list of cryptos that should not be considered for trading due to
  - trading volume below certain threshold
  - too many no trade breaks indicating  a limited number of market participants
- trading mode
  - **test mode** uses a known market data subset from the past and a fixed set of cryptos as regression test set
  - **simulation** considers the configured black list of cryptos, the dynamic set of tradable cryptos, current market data but is not actively trading
  - **production** considers the configured black list of cryptos, the dynamic set of tradable cryptos, current market data and is actively trading

### state machines

There are the following state machines that influence each other:

- states of asset portfolio
- states of tradable cryptos

## analysis

## test

## Tasks and priority

- unclear why there are so few buy signals
  - compare catalyst data with ccxt data
  - use catalyst data to simulate trading
  - analyse with visualization
- optimize high frequent trading to execute buy or sell signal
  - today rule based - ML better?
- Scilab.learn to adapt linear

## Currency machine learning trading strategy

### Classifiers (idea Dec 2018)

- Distinguish between
individual currency performance to decide about sell or buy → CPC
- provide estimation about best volume distribution of own assets among currency candidates → VDC
- provide estimation about best next trading step in sub-minute timeframe → DTC

### Currency performance classifier (CPC)

#### CPC Objective

- provide estimation about currency performance
- not sure whether common or separate CPC per currency is required
  - common —> more training material
  - separate —> specific per currency

CPC time aggregation classifier target is % price performance either
without time limitation but limitation of intermediate loss  or
without intermediate loss but within certain time frame (preferred as peaks down are hard to foresee)

#### CPC Features

As the absolute value is not crucial for the pattern the % change of high, low, open(redundant with previous close), close in relation to the last close price is used.

- (current close - prev close) / prev close → % delta between candles = D
- (high-low) / current close → % candle height = H
- (high - max(open, close)) / (high-low) → % candle top = T
- (min(open, close) - low) / (high-low) → % candle bottom = B

- For volume the % in relation to the average of considered feature time period is used = V

##### DHTBV features labels

- buy: > 1% performance without intermediate loss is the minimum to reach.
  - earning: price performance >0,5%
  - 2 x 0,1% trading fees to buy and sell.
  - Consider buy and sell slippage of 0,1% each - better to consider the real slippage of one minute bar for labeling
  - to avoid too high frequent buy and sell >=1% should be the target to invest
- sell: < -0,2%
  - that is what needs to be paid for fees.
  - In reality additional slippage of one period for buy and sell
sell
  - loss can be taken into account by just looking one period ahead after -0,2% is reached

Considering that binary classifiers perform best, those are used for different aggregations:

#### CPC classifier 4 hours

- features: DHTBV of last 10 4-hour periods, MA(7), MA(25), MA(99)
- regression estimate target: measured price performance without intermediate loss
- buy target: price performance >=1% without intermediate loss
- sell target: price performance < 0%

#### CPC classifier 1 hour

- features: DHTBV of last 10 hours, MA(7), MA(25)
- regression estimate target: measured price performance without intermediate loss
- buy target: price performance >=1% without intermediate loss
- sell target: price performance < 0%

#### CPC classifier 15 minutes

- features: DHTBV of last 10 15-minutes periods, MA(7), MA(25)
- regression estimate target: measured price performance without intermediate loss
- buy target: price performance >=1% without intermediate loss
- sell target: price performance < 0%

#### CPC classifier 5 minutes

- features: DHTBV of last 10 5-minutes periods, MA(7), MA(25)
- regression estimate target: measured price performance without intermediate loss
- buy target: price performance >=1% without intermediate loss
- sell target: price performance < 0%

#### CPC classifier 1 minute

- features: DHTBV of last 10 1-minute periods, MA(7), (last minute DHTBV of BTC/USDT —> first check that BTC is indeed front runner)
- regression estimate target: measured price performance without intermediate loss
- buy target: price performance >=1% without intermediate loss
- sell target: price performance < 0%

#### CPC regression estimate

- features: results from all CPC regression estimators above
- regression estimate target: measured price performance without intermediate loss

#### CPC buy/hold/sell classifier

- features: results from all CPC buy and sell time aggregation classifiers above
- buy target: buy signal from valid buy signal combination with best performance
- sell target: sell signal from valid buy signal combination with best performance
- buy target: buy signal from any of the CPC buy time aggregation classifiers and no sell indication from any of the CPC sell time aggregation classifiers with shorter time period than the buy signal CPC
- sell target: sell signal from any of the CPC sell time aggregation classifiers and no buy indication from any of the CPC buy time aggregation classifiers with shorter time period than the sell signal CPC
- hold target: if neither buy nor sell target
- [ ] TODO open issue: targets are set by intuition, it would be better to measure them

  - what to achieve? maximum earnings by using volatility
  - upper bound earnings indicator:
  - measure per currency the average performance per day
  - measure overall performance per day considering that parallel improvements only count once
  - measure overall performance by considering best case
  - measure overall performance by considering averaged case
  - measure overall performance by considering worst case

### Volume distribution classifier (VDC) - current

provide estimation about best volume distribution of own assets among currency candidates (multi class classifier)

- features:
CPC regression estimator results of all currency candidates
% 15-minutes volume in relation to 10 x 4 hour average 15-minutes volume
- target: CPCs that outperform others within 15 minutes
  - only those that improve >1% at all else 0
  - those that are in the highest 1% price performance buckets, i.e. >1%, >2%, >3%, >4%, >5%
- currency candidate shall qualify by
  - liquidity (current volume traded)
  - traded volume per minute averaged over last 40 h > 2 x volume to invest to be able to react on minute basis
  - shall have a CPC buy signal

#### Volume distribution classifier (VDC)

Nov 2018 → review

#### VDC Objective

- Only move asset allocation if the estimation is reliable and the change makes economically sense.
- Classify the performance estimation results and estimate the estimation reliability across all assets.
- Distribute asset allocation accordingly.
- Any change need to be better than 0,4% considering 0,2% sell fee and 0,2% buy fee compared against no change

##### VDC Feature vector

- Concatenation of 18 Binance USDT currencies without PAX and TUSD with
CPC result vector (9 elements)
- trading volume per minute averaged over the last 5 minutes
- own assets currently allocated to the specific currency in % of all own assets
- target allocation of own assets in % of all own assets
=12*18=216 elements

##### VDC result vector (19 elements)

- Concatenation of 18 Binance USDT currencies without USDT, PAX and TUSD
- supervisor function:
  - 1 if target change really materialized as estimated
  - -0,2 for every level lower
  - 0 otherwise
- strong buy 5%, if >= 5% price performance without in between loss of >=0,2%

Calculate asset allocation to currency (reconsider)

- explanation to reconsider: in case of steep price performances reshuffling shall be timely. Consider hysteresis to avoid frequent changes that don’t cover their fees.

Don’t move asset distribution when expected improvements have not yet materialized as fees need to be paid but move if asset become available due to sell signal or if predictions don’t materialize

Calculated volume change

From best to worse distribute volume among the best rated currencies as target

- delta = target - current
- if delta > minute volume averaged over last 5 minutes then reduce target and delta to last minute volume averaged over last 5 minutes
- add the reduction to all other best rated currencies or if all best rated currencies already have a calculated volume change then include the next lower performance category and distribute the reduction among them in the same way

### Dynamics trading classifier (DTC) - current

Provide estimation about best next trading step in sub-minute timeframe.

- Input is amount to sell or to buy (2 different DTC classifiers)
- results: amount % of volume to sell/buy and % price in relation to last minute close
- features: order book quantiles (10 ask, 10 buy) covering in 10% volume steps the % price in relation to last minute close, HLOCV of last 1-minute period
  - consumes CPC results of specific currency as input
  - consumes VDC volume result of specific currency as input

- not sure whether common or separate DTC for buy and sell side is better → decision separate classifiers
- not sure DTC per currency is required (I don’t assume so) → decision common of all currencies in step 1

### Dynamics trading classifier (DTC) - Nov 2018 → review

#### Preparations

- All prices in % of last minute OCHL average price to become robust against absolute prices
- All volume in % of volume to be bought or sold to average to become robust against absolute volume

#### Feature vector

- CPC result vector (9 elements)
- bid and ask prices[%] after 10%, 20%, 30%, 40%, 50%, 60%, 70% 80, 90% of vol to be sold or bought
- volume of last minute
- minute volume of 5 last minutes averaged
- last 20 trades with price and volume sorted by price

#### Classifier result vector

- maker 10%, 20%, 30%, 40%, 50%, 60% 70% 80% 90% 100%, if volume sold in that bucket and improve <=0,2%
- taker 10%, 20%, 30%, 40%, 50%, 60% 70% 80% 90% 100%, if volume sold in that bucket and improve <=0,2%
- wait, if situation improves >0,2%
- use the max of those and in doubt taker with more %

### Currency performance classifier (CPC) - Nov 2018 → outdated

Explanation for outdated: too many features; furthermore preference for binary classifiers with superior performance

#### Preparations - Nov 2018 → outdated

- all prices in % of 10x 1 week Open Close High Low (OCHL) average to become robust against absolute prices
- all volume in % of 10x 1 week volume average to become robust against absolute volume

#### Feature vector - Nov 2018 → outdated

- 10x 1min OHCL Vol ==> 10min total coverage
- 10x 5min OHCL Vol skipping 2 most recent intervals ==> 1h total coverage
- 6x 30min OCHL Vol skipping 2 most recent intervals ==> 4h total coverage
- 11x 4h OCHL Vol skipping 1 most recent interval ==> 48h=2d total coverage
- 5x 1d OCHL Vol skipping 2 most recent intervals ==> 7d=1wk total coverage
- 10x 1wk OCHL Vol skipping 1 most recent interval ==> 11wks total coverage
- 5x 1min OHCL BTC Vol ==> most recent BTC moves
- = (10+10+6+11+5+10+5)*5 elements = 285 elements

#### CPC result vector (9 elements)

- strong buy 5%, if >= 5% price performance without in between loss of >=0,2%
- strong buy 4%, if >= 4% price performance without in between loss of >=0,2%
- strong buy 3%, if >= 3% price performance without in between loss of >=0,2%
- strong buy 2%, if >= 2% price performance without in between loss of >=0,2%
- strong buy 1%, if >= 1% price performance without in between loss of >=0,2%
- buy, if >= 1% price performance without in between loss of >=0,5%
- hold, if >= 1% price performance without in between loss of >=0,8%
- strong sell, if loss >1%
- sell, if worse than hold and better than strong sell

The module hierarchie:

- `env_config` provides the computing and usage environment. Changing from production to test or from a linux to a floydhub or colab environment shall only be switched here and shall have no functional impact to the rest of the code.
- `local_xch` is the local exchange abstraction layer that either provides live or historic data.
- `crypto_target_features` provides the feature and target class calculation. It shall abstract from the origin of the data, i.e. whether it is live data or historic data. classes can only be calculated when future data vectors are available to calculate the target class.
- `history_sets` provides training, validation and test data.
- `classify_keras` provide MLP neural nets with training, evaluation and classification (e.g. for production usage) methods.
- `trading` provdes live trading methods

## Notes

Conda, pip, ipython, jupyter

conda is a package manager not bound to a language but an environment structure that maintains dependencies. conda distinguishes packages installed and environments. Enabling cohabitation of various library and language versions.

pip is a python package manager that is specific to python but can integrated into different environments. It does not follow up on non-python dependencies.

ipython is an interactive interpreter environment above python. It provides magical single line % and cell %% commands. Jupyter notebooks is a spin-off of ipython.

Jupyter notebooks support interactive computing environment for different programming languages (although created initially for python)

## To do

- ActiveFeatures class shall be bound to the classifier and loaded with it
- Use trading from interactive dashboard
- limit bookkeeping safe_cache to 2* chs.ActiveFeatures.history() to avoid unnecessary large files
