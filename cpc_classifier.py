#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 18:40:09 2019

@author: tc

"""

from __future__ import print_function

import os
# import pandas as pd
import targets_features as t_f

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


# MODEL_PATH = '/Users/tc/tf_models/crypto' # local execution
MODEL_PATH = '/content/gdrive/My Drive/tf_models/crypto' # Colab execution

#cur_cand = ['xrp_usdt', 'btc_usdt']
CUR_PAIR = 'xrp_usdt'
VAL_RATIO = 0.1
tf_vectors = None

def load_targets_features(currency_pair: str):
    """converts historic catalyst data of currency pairs into classifier target and feature vectors
    and stores them for classifier training and evaluation
    """
    fname = t_f.DATA_PATH + '/' + currency_pair + '.msg'
    tf_vec = t_f.TfVectors(filename=fname)
    assert tf_vec is not None, "missing tf vectors from {fname}"
    return tf_vec



def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear classification model.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(7000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels



def train_linear_classifier_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear classification model.

  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.

  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns
      to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column
      to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns
       to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column
       to use as target for validation.

  Returns:
    A `LinearClassifier` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods

  # Create a linear classifier object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_classifier = tf.estimator.LinearClassifier(
      feature_columns=construct_feature_columns(training_examples),
      optimizer=my_optimizer, model_dir=MODEL_PATH
  )
  print(f"model directory: {linear_classifier.model_dir}")
  # Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples,
                                          training_targets,
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                  training_targets,
                                                  num_epochs=1,
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                    validation_targets,
                                                    num_epochs=1,
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("LogLoss (on training data):")
  training_log_losses = []
  validation_log_losses = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_classifier.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
    training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

    validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
    validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

    training_log_loss = metrics.log_loss(training_targets, training_probabilities)
    validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_log_loss))
    # Add the loss metrics from this period to our list.
    training_log_losses.append(training_log_loss)
    validation_log_losses.append(validation_log_loss)
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel("LogLoss")
  plt.xlabel("Periods")
  plt.title("LogLoss vs. Periods")
  plt.tight_layout()
  plt.plot(training_log_losses, label="training")
  plt.plot(validation_log_losses, label="validation")
  plt.legend()

  return linear_classifier


def execute():
    tf_vectors = load_targets_features(CUR_PAIR)
    set_dict = tf_vectors.split_data(1)
    print(set_dict['training'].describe())
    print(set_dict['validation'].describe())
    print(set_dict['test'].describe())
    c_df = tf_vectors.vec(1) # focus on the T1 aggregation
    val_count = math.ceil(len(c_df) * VAL_RATIO) # number of validation samples
    train_count = math.floor(len(c_df) * (1-VAL_RATIO)) # nbr of training samples
    f_list = list(c_df.columns) # column names of features
    f_list.remove('close')
    f_list.remove('buy')
    f_list.remove('sell')
    f_df = c_df[f_list] # features dataframe
    b_df = c_df['buy'] # buy target dataframe
    s_df = c_df['sell'] # sell target dataframe
    f_train_df = f_df.head(train_count) # features training dataframe
    b_tgt_train_df = b_df.head(train_count) # buy target training dataframe
    s_tgt_train_df = b_df.head(train_count) # sell target training dataframe
    f_val_df = f_df.tail(val_count) # features validation dataframe
    b_tgt_val_df = b_df.tail(val_count) # buy validation dataframe
    s_tgt_val_df = s_df.tail(val_count) # sell validation dataframe


    b_linear_classifier = train_linear_classifier_model(
        learning_rate=0.000005,
        steps=500,
        batch_size=20,
        training_examples=f_train_df,
        training_targets=b_tgt_train_df,
        validation_examples=f_val_df,
        validation_targets=b_tgt_val_df)


