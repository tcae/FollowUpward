import numpy as np
# from sklearn.linear_model import LinearRegression
import condensed_features as cof


def time_linear_regression(df):
    """ Receives a one column data dataframe with a Datetimeindex in fixed minute frequency.

        Returns a (hourly_delta, regression_df) tuple containing the regression delta within 1 hour
        and a two element y coordinate array (first and last point) of the data regression line.
    """
    # X = np.arange(len(df)).reshape(-1, 1)
    Y = df.values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    # linear_regressor = LinearRegression()  # create object for the class
    # linear_regressor.fit(X, Y)  # perform linear regression
    # Y_pred = linear_regressor.predict(X[[0, -1]])[:, 0]  # sufficient to predict 1st and last X points

    grad_arr, yend_arr = cof.regression_line(Y, len(df))
    hourly_grad = grad_arr[-1] * 60
    yend = yend_arr[-1]
    ystart = yend - grad_arr[-1] * (len(df)-1)
    y_regr = np.array([ystart, yend])
    # delta = Y_pred[-1] - Y_pred[0]
    # timediff_factor = 60 / (len(Y) - 1)  # constraint: Y values have consecutive 1 minute distance
    # delta = delta * timediff_factor
    # print(f"delta-Y: {Y_pred - y_regr} = Y_pred{Y_pred} - yend {y_regr}, grad diff {delta-hourly_grad} ")

    return (hourly_grad, y_regr)
