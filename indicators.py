import numpy as np
from sklearn.linear_model import LinearRegression


def time_linear_regression(df):
    """ Receives a one column data dataframe with a Datetimeindex in fixed frequency.

        Returns a (hourly_delta, regression_df) tuple containing the regression delta within 1 hour
        and a one dimensional data regression df with the same Datetimeindex as the input df.
    """
    X = np.arange(len(df)).reshape(-1, 1)
    Y = df.values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X[[0, -1]])[:, 0]  # sufficient to predict 1st and last X points
    # df = df.to_frame()
    # df = df.assign(Y_regression=Y_pred)
    # df["delta"] = (df.Y_regression - df.Y_regression.shift(1))
    # print("df.Y_reg", df.head(5))
    delta = Y_pred[-1] - Y_pred[0]
    timediff_factor = 60 / (len(Y) - 1)  # constraint: Y values have consecutive 1 minute distance
    # timediff_factor = float(pd.Timedelta(60, unit="m") / pd.Timedelta(df.index[1]-df.index[0]))
    delta = delta * timediff_factor
    return (delta, Y_pred)
