# FollowUpward

Follow crypto trend - a python and ML learning project (documented in [Markdown][MD])

[MD]: https://daringfireball.net/projects/markdown/dingus


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
- pickle and pandas hdf5 for storage of results
- Dash and plotly for interactive visualizations
- development and debug environment based on Visual Studio Code (Jupyter notebooks are not suited for easy debugging)
- conda for package consistency and virtual environment

### package installations of environment tf3 (slower than tf5)

- conda install -c intel intelpython3_full
- conda install dash
- pip install dash_colorscales
- pip install ccxt
- pip install tables

|Name                     | Version           |        Build  | Channel
|-------------------------|-------------------|---------------|-----------
|tables                   | 3.6.1             |       pypi_0  | pypi
|ccxt                     | 1.24.20           |       pypi_0  | pypi
|dash                     | 1.9.1             |        py_0   | conda-forge
|dash-core-components     | 1.8.1             |        py_0   | conda-forge
|dash-html-components     | 1.0.2             |        py_0   | conda-forge
|dash-renderer            | 1.2.4             |        py_0   | conda-forge
|dash-table               | 4.6.1             |        py_0   | conda-forge
|intelpython3_full        | 2020.0            |           0   | intel

Downgrade package `gast` to version 0.2.2 with `conda install gast=0.2.2`

### package installationsof environment tf5

- pip install tensorflow-transform
- pip install ccxt
- pip install tables
- conda install dash
- conda install -c conda-forge pandas
- conda install requests  # this package is not yet used but may replace ccxt in the future to use binance streaming
- pip install bottleneck  # version 1.3.2  - not yet used but may be used for rolling window calculation of 1D regression

With tensorflow-transform the following packages are installed: absl-py-0.8.1 apache-beam-2.19.0 astor-0.8.1 avro-python3-1.9.2.1 cachetools-3.1.1 chardet-3.0.4 crcmod-1.7 dill-0.3.1.1 docopt-0.6.2 fastavro-0.21.24 fasteners-0.15 future-0.18.2 gast-0.2.2 google-api-core-1.16.0 google-api-python-client-1.8.0 google-apitools-0.5.28 google-auth-1.13.1 google-auth-httplib2-0.0.3 google-auth-oauthlib-0.4.1 google-cloud-bigquery-1.17.1 google-cloud-bigtable-1.0.0 google-cloud-core-1.3.0 google-cloud-datastore-1.7.4 google-cloud-pubsub-1.0.2 google-pasta-0.2.0 google-resumable-media-0.4.1 googleapis-common-protos-1.51.0 grpc-google-iam-v1-0.12.3 grpcio-1.28.1 h5py-2.10.0 hdfs-2.5.8 httplib2-0.12.0 idna-2.9 keras-applications-1.0.8 keras-preprocessing-1.1.0 markdown-3.2.1 mock-2.0.0 monotonic-1.5 numpy-1.18.2 oauth2client-3.0.0 oauthlib-3.1.0 opt-einsum-3.2.0 pbr-5.4.5 protobuf-3.11.3 pyarrow-0.15.1 pyasn1-0.4.8 pyasn1-modules-0.2.8 pydot-1.4.1 pymongo-3.10.1 pyparsing-2.4.7 python-dateutil-2.8.1 pytz-2019.3 requests-2.23.0 requests-oauthlib-1.3.0 rsa-4.0 scipy-1.4.1 six-1.14.0 tensorboard-2.1.1 tensorflow-2.1.0 tensorflow-estimator-2.1.0 tensorflow-metadata-0.21.1 tensorflow-serving-api-2.1.0 tensorflow-transform-0.21.2 termcolor-1.1.0 tfx-bsl-0.21.4 typing-3.7.4.1 typing-extensions-3.7.4.2 uritemplate-3.0.1 urllib3-1.25.8 werkzeug-1.0.1 wrapt-1.12.1

With ccxt the following packages are installed: installs aiodns-1.1.1 aiohttp-3.6.2 async-timeout-3.0.1 attrs-19.3.0 ccxt-1.26.5 cffi-1.14.0 cryptography-2.9 multidict-4.7.5 pycares-3.1.1 pycparser-2.20 yarl-1.1.0

With tables the following packages are installed numexpr-2.7.1 tables-3.6.1

With dash the following packages are installed:
  click              conda-forge/noarch::click-7.1.1-pyh8c360ce_0
  dash               conda-forge/noarch::dash-1.9.1-py_0
  dash-core-compone~ conda-forge/noarch::dash-core-components-1.8.1-py_0
  dash-html-compone~ conda-forge/noarch::dash-html-components-1.0.2-py_0
  dash-renderer      conda-forge/noarch::dash-renderer-1.2.4-py_0
  dash-table         conda-forge/noarch::dash-table-4.6.1-py_0
  flask              conda-forge/noarch::flask-1.1.2-pyh9f0ad1d_0
  flask-compress     conda-forge/noarch::flask-compress-1.4.0-py_0
  future             conda-forge/linux-64::future-0.18.2-py37_0
  itsdangerous       conda-forge/noarch::itsdangerous-1.1.0-py_0
  jinja2             conda-forge/noarch::jinja2-2.11.1-py_0
  markupsafe         conda-forge/linux-64::markupsafe-1.1.1-py37h516909a_0
  plotly             conda-forge/noarch::plotly-4.6.0-pyh9f0ad1d_0
  pyyaml             intel/linux-64::pyyaml-5.3-py37_0
  retrying           conda-forge/noarch::retrying-1.3.3-py_2
  six                intel/linux-64::six-1.13.0-py37_2
  werkzeug           conda-forge/noarch::werkzeug-1.0.1-pyh9f0ad1d_0
  yaml               intel/linux-64::yaml-0.1.7-5

With pandas the following packages are installed:
  _libgcc_mutex      conda-forge/linux-64::_libgcc_mutex-0.1-conda_forge
  _openmp_mutex      conda-forge/linux-64::_openmp_mutex-4.5-1_llvm
  libblas            conda-forge/linux-64::libblas-3.8.0-16_openblas
  libcblas           conda-forge/linux-64::libcblas-3.8.0-16_openblas
  libgfortran-ng     conda-forge/linux-64::libgfortran-ng-7.3.0-hdf63c60_5
  liblapack          conda-forge/linux-64::liblapack-3.8.0-16_openblas
  libopenblas        conda-forge/linux-64::libopenblas-0.3.9-h5ec1e0e_0
  llvm-openmp        conda-forge/linux-64::llvm-openmp-10.0.0-hc9558a2_0
  numpy              conda-forge/linux-64::numpy-1.18.1-py37h8960a57_1
  pandas             conda-forge/linux-64::pandas-1.0.3-py37h0da4684_0
  python-dateutil    conda-forge/noarch::python-dateutil-2.8.1-py_0
  python_abi         conda-forge/linux-64::python_abi-3.7-1_cp37m
  pytz               conda-forge/noarch::pytz-2019.3-py_0

With requests the followings packages are installed:
  asn1crypto         intel/linux-64::asn1crypto-1.3.0-py37_0
  cffi               intel/linux-64::cffi-1.13.2-py37_2
  chardet            intel/linux-64::chardet-3.0.4-py37_3
  cryptography       intel/linux-64::cryptography-2.8-py37_0
  idna               intel/linux-64::idna-2.8-py37_0
  pycparser          intel/linux-64::pycparser-2.19-py37_1
  pyopenssl          intel/linux-64::pyopenssl-19.1.0-py37_1
  pysocks            intel/linux-64::pysocks-1.7.0-py37_1
  requests           intel/linux-64::requests-2.22.0-py37_1
  urllib3            intel/linux-64::urllib3-1.25.7-py37_0

## Main code entry points

- `adapt_eval` (earlier `classify_keras` - now deprecated) is used to train and evaluate ML.
- `trading` is used to trade via ccxt.
- `training_dashboard` are for visual analysis
- `update_crypto_history` updates the cached historic ohlcv and feature data

## Design approach

In order to focus on the ML part data handling shall be unified over all data producers.

- `env_config.py` provides all environment switches and common utilities mainly through the Env class
- `CryptoData()` in `cached_crypto_data.py` provides common data handling
- `Ohlcv(CryptoData)` in `cached_crypto_data.py` provides ohlcv (open, high, low, close, volume) data provided by Binance via ccxt
- `Features(CryptoData)` in `cached_crypto_data.py` is common superclass for all features and is enforcing cache usage
- `F3cond14(ccd.Features)` in `condensed_features.py`: features based on normalized 1D regression
- `F1agg110(ccd.Features)` in `aggregated_features.py`: features based on normalized ohlcv data aggregation
- `T10up5low30min(Targets)` in `crypto_targets.py`: targets with buy when more than 1% and sell when -0.5% within given time period - in between hold. The hysteresis shall avoid loss causing high frequent trading
- `PredictionData(ccd.CryptoData)` in `prediction_data.py`: calculates predictioins of a classifier
- `PerformanceData(ccd.CryptoData)` in `prediction_data.py`: calculates the %performance of a classifier
- `TrainingData()` in `adaptation_data.py` generates seperates the cached history data in distinct training, validation, test data sets based on a configuration csv file
- `PredictionData(ccd.CryptoData)` in `prediction_data.py` is a common prediction superclass
- `Classifier(Predictor)` in `classifier_predictor.py` is a classifier that uses scikit_learn scaler for feature normalization and keras models for machine learning
- `TrainingGenerator(keras.utils.Sequence)` in `adaptation_data` provides a multi processing and threat save generator for keras model adaptation
- `ValidationGenerator(keras.utils.Sequence)` in `adaptation_data` provides a multi processing and threat save generator for keras model evaluation
- `AssessmentGenerator(keras.utils.Sequence)` in `adaptation_data` provides a multi processing and threat save generator for performance assessment
- `Xch()` in `local_xch.py` provides encapsulation to ccxt, e.g. checking limits, exception handling
- `Bk()` in `bookkeeping.py` keeps track of actions, currencies under monitoring, and trades using `Xch()`
- `Trading()` in `trading.py` provides the core trading loop using `Bk()`
- `trading_dashboard.py` provides a visualization of historic data, features, targets based on Dash
- `indicators.py` provides 1D regression

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

## Ideas and Observations

### Design ideas

CryptoData shall be extended to store and get data quick and easy but there shall be a layer in between to enable classes using the same data for different things, e.g.

- update the values periodically in own process and read only file
- Gestaltklassen targets
- use distance to high and low of aggregations 30min, 1h, 2h, 4h, 12h,
- features normalized, capped, masked (use some for visualization and others for ML)
- ohlcv gaps stored but without gaps used
- target classes or regression values

### Targets determination

Problem: when to decide a temporary fall back shall be ignores and when not?

A buy signal shall only be given if there is a positive observable trend. Respectively, a sell target on a negative observable trend. Observability can come with different smooth factors, e.g. number of minutes to calculate a regression line. Situations may change that a trade target should follow a less smoothend trend or a smoother trend. To determine that one can start with the lowest smooth factor (e.g. just 1 minute aggragate) and compile the gain of a positive trend. If a sel buy differnece is less than a sell threshold and the next smoother trend compiles more gain then the target label generation follows the smoother trend instead of the less smooth trend, which is the default.

Problem: adequate loss function

Calculate the missed compiled gain or realized compiled loss as loss function - not only the gain/loss between consecutive samples because we want to weight in the beginning of a gain phase heavier than at its end. This also penelaizes HOLD less than a trade signal.

Problem: Target metrics required

- correlation of gain/loss with high/low distance
- correlation of gradient sign change with minute distance (are there patterns that correlate with regression minutes?)
- correlation of single gain with minutes of aggregation


### Oberservations

Gradient alone may be insufficient - gain between gradient sign changes may be significant, e.g. mid term gaining crypto shows small losses between gradient sign changes but big gains. Hence the last complete loss and the last complete gain per aggregation may add value.

Which aggregation is most profitable? Depends on last observed gains/losses, the liquidity (do I find a counterpart at the price I want to trade?)

A price that builds up creates a counterforce to cash the gains and a falling price builds a counterforce because the likelyhood of correction increases. That counterforce becomes stronger the shorter time the change needs and the higher the change amplitude. Thus it may be characterized by change * time window of change.
The last 2 counterforces (one up, one down) shall be considerd because a build up counterforce may compensate the force build up