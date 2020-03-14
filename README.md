# FollowUpward

Follow crypto trend - a python and ML learning project

## Vision

A development environment that supports engineering features and classifier nets crypto trading decisions.
Stock trading is a possible future step but cryoto trading offers more opportunities (and risks) due to higher volatility.

## Environment

- python 3.7.3
- pandas and numpy for data handling and calculations outside model adaptation
- scikit learn for scaler adaptation
- keras for ML models
- Tensorflow 2 servings Keras for machine learning.
- ccxt for crypto trading.
- talos for hyperparameter optimization
- pickle and pandas hdf5 for storage of results
- Dash and plotly for interactive visualizations
- development and debug environment based on Visual Studio Code (Jupyter notebooks are not suited for easy debugging)
- conda for package consistency and virtual environment

## Main code entry points

- `prediction_data` (earlier `classify_keras` - now deprecated) is used to train and evaluate ML.
- `trading` is used to trade via ccxt.
- `training_dashboard` are for visual analysis
- `update_crypto_history` updates the cached historic ohlcv and feature data

## Design approach

In order to focus on the ML part data handling shall be unified over all data producers.

- `env_config.py` provides all environment switches and common utilities mainly through the Env class
- `CryptoData()` in `cached_crypto_data.py` provides common data handling
- `Ohlcv(CryptoData)` in `cached_crypto_data.py` provides ohlcv (open, high, low, close, volume) data provided by Binance via ccxt
- `Features(CryptoData)` in `cached_crypto_data.py` is common superclass for all features and is enforcing cache usage
- `F2cond20(ccd.Features)` in `condensed_features.py`: features based on normalized 1D regression
- `F1agg110(ccd.Features)` in `aggregated_features.py`: features based on normalized ohlcv data aggregation
- `T10up5low30min(Targets)` in `crypto_targets.py`: targets with buy when more than 1% and sell when -0.5% within given time period - in between hold. The hysteresis shall avoid loss causing high frequent trading
- `PredictionData(ccd.CryptoData)` in `prediction_data.py`: calculates predictioins of a classifier
- `PerformanceData(ccd.CryptoData)` in `prediction_data.py`: calculates the %performance of a classifier
- `TrainingData()` in `adaptation_data.py` generates seperates the cached history data in distinct training, validation, test data sets based on a configuration csv file
- `PredictionData(ccd.CryptoData)` in `prediction_data.py` is a common prediction superclass
- `Classifier(Predictor)` in `prediction_data.py` is a classifier that uses scikit_learn scaler for feature normalization and keras models for machine learning
- `TrainingGenerator(keras.utils.Sequence)` in `adaptation_data` provides a multi processing and threat save generator for keras model adaptation
- `ValidationGenerator(keras.utils.Sequence)` in `adaptation_data` provides a multi processing and threat save generator for keras model evaluation
- `AssessmentGenerator(keras.utils.Sequence)` in `adaptation_data` provides a multi processing and threat save generator for performance assessment
- `Xch()` in `local_xch.py` provides encapsulation to ccxt, e.g. checking limits, exception handling
- `Bk()` in `bookkeeping.py` keeps track of actions, currencies under monitoring, and trades using `Xch()`
- `Trading()` in `trading.py` provides the core trading loop using `Bk()`
- `trading_dashboard.py` provides a visualization of historic data, features, targets based on Dash
- `indicators.py` provides 1D regression
_
## Deprecated
- `classify_keras.py` that provided in the past the adaptaion
- `Catalyst2Pandas.ipynb` is a jupyter notebook that provided historic Catalyst data. Required old pandas version. Is replaced by ccxt.
- `crypto_history_sets.py` is the former management of training, validation and test sets - was too complex and replced by the CryptoData and TrainingData approach

`history2pandas.py` to load historical Catalyst data into pandas stored as msgpack files.
//: # `Catalyst2PandasData.ipynb` to load historical Catalyst data into pandas stored as msgpack files.


### Configurations

- black list of cryptos that should not be considered for trading due to
  - trading volume below certain threshold
  - too many no trade breaks indicating  a limited number of market participants
