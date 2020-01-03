import indicators as ind

""" Alternative feature set. Approach: less features and closer to intuitive performance correlation
    (what would I look at?).

    - regression line percentage increase per hour calculated on last 10d basis
    - regression line percentage increase per hour calculated on last 12h basis
    - regression line percentage increase per hour calculated on last 4h basis
    - regression line percentage increase per hour calculated on last 0.5h basis
    - regression line percentage increase per hour calculated on last 5min basis for last 2x5min periods
    - percentage of last 5min mean volume compared to last 1h mean 5min volume
    - not directly used: SDU = absolute standard deviation of price points above regression line
    - SDU - (current price - regression price) (== up potential) for 12h, 4h, 0.5h, 5min regression
    - not directly used: SDD = absolute standard deviation of price points below regression line
    - SDD + (current price - regression price) (== down potential) for 12h, 4h, 0.5h, 5min regression
"""


def calc_features(minute_data):
    vec = None
    return vec
