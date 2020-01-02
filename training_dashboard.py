import json
from textwrap import dedent

import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# import plotly.graph_objs as go
import pandas as pd
import numpy as np
from env_config import Env
# import env_config as env
import cached_crypto_data as ccd
import indicators as ind

from sklearn.linear_model import LinearRegression

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
cdf = None
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}


app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Checklist(
                id='crossfilter-crypto-select',
                options=[{'label': i, 'value': i} for i in Env.usage.bases],
                labelStyle={'display': 'inline-block'}
            ),
            html.Button('all', id='all-button', style={'display': 'inline-block'}),
            html.Button('none', id='none-button', style={'display': 'inline-block'}),
        ], style={'display': 'inline-block'}),
        # html.H1("Crypto Price"),  # style={'textAlign': "center"},
        dcc.Graph(id="full-day-time-graph"),
        dcc.Graph(id="ten-day-time-graph"),
        dcc.Graph(id="halfyear-graph"),
        dcc.Graph(id="full-time-graph"),
        html.Div(className='row', children=[
            html.Div([
                dcc.Markdown(dedent("""
                    **Hover Data**

                    Mouse over values in the graph.
                """)),
                html.Pre(id='hover-data', style=styles['pre'])
            ], className='three columns'),

            html.Div([
                dcc.Markdown(dedent("""
                    **Click Data**

                    Click on points in the graph.
                """)),
                html.Pre(id='click-data', style=styles['pre']),
            ], className='three columns'),

            html.Div([
                dcc.Markdown(dedent("""
                    **Selection Data**

                    Choose the lasso or rectangle tool in the graph's menu
                    bar and then select points in the graph.

                    Note that if `layout.clickmode = 'event+select'`, selection data also
                    accumulates (or un-accumulates) selected data if you hold down the shift
                    button while clicking.
                """)),
                html.Pre(id='selected-data', style=styles['pre']),
            ], className='three columns'),

            html.Div([
                dcc.Markdown(dedent("""
                    **Zoom and Relayout Data**

                    Click and drag on the graph to zoom or click on the zoom
                    buttons in the graph's menu bar.
                    Clicking on legend items will also fire
                    this event.
                """)),
                html.Pre(id='relayout-data', style=styles['pre']),
            ], className='three columns')
        ]),
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px',
        'width': '49%',
        'display': 'inline-block'
    }),

    html.Div([
        dcc.RadioItems(
            id='crossfilter-crypto-radio',
            options=[{'label': i, 'value': i} for i in Env.usage.bases],
            labelStyle={'display': 'inline-block'}
        ),
        dcc.Checklist(
            id='crossfilter-indicator-select',
            options=[{'label': i, 'value': i} for i in ind.options],
            value=[],
            labelStyle={'display': 'inline-block'}
        ),
        dcc.Graph(id='zoom-in-graph'),
        # dcc.Graph(id='volume-signals-graph'),
    ], style={'display': 'inline-block', 'float': 'right', 'width': '49%'}),

])


@app.callback(
    dash.dependencies.Output('full-time-graph', 'figure'),
    [dash.dependencies.Input('crossfilter-crypto-select', 'value')])
def update_graph(bases):
    graph_bases = []
    if bases is not None:
        for base in bases:
            dcdf = cdf[base].resample("D").agg({"open": "first"})
            # dcdf = cdf[base].resample("D").agg({"open": "first", "close": "last", "high": "max",
            #                                     "low": "min", "volume": "sum"})
            dcdf = dcdf/dcdf.max()  # normalize
            graph_bases.append(dict(
                x=dcdf.index,
                y=dcdf["open"],
                mode='lines',  # 'lines+markers'
                name=base
            ))
    return {
        'data': graph_bases,
        'layout': {
            'height': 450,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': "normalized crypto prices"
            }],
            'yaxis': {'type': 'linear'},
            'xaxis': {'showgrid': False}
        }
    }


@app.callback(
    Output('crossfilter-crypto-select', 'value'),
    [Input('all-button', 'n_clicks_timestamp'), Input('none-button', 'n_clicks_timestamp')])
def select_all_none(all_click_ts, none_click_ts):
    if (all_click_ts is not None) and (none_click_ts is not None):
        if all_click_ts > none_click_ts:
            return Env.usage.bases
        else:
            return []
    if all_click_ts is not None:
        return Env.usage.bases
    if none_click_ts is not None:
        return []
    return [Env.usage.bases[0]]


@app.callback(
    Output('crossfilter-crypto-radio', 'value'),
    [Input('crossfilter-crypto-select', 'value')])
def radio_range(selected_cryptos):
    if (selected_cryptos is not None) and (len(selected_cryptos) > 0):
        return selected_cryptos[0]
    else:
        return Env.usage.bases[0]


@app.callback(
    Output('hover-data', 'children'),
    [Input('full-time-graph', 'hoverData')])
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)


@app.callback(
    Output('click-data', 'children'),
    [Input('full-time-graph', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)


@app.callback(
    Output('selected-data', 'children'),
    [Input('full-time-graph', 'selectedData')])
def display_selected_data(selectedData):
    return json.dumps(selectedData, indent=2)


@app.callback(
    Output('relayout-data', 'children'),
    [Input('full-time-graph', 'relayoutData')])
def display_relayout_data(relayoutData):
    return json.dumps(relayoutData, indent=2)


@app.callback(
    dash.dependencies.Output('halfyear-graph', 'figure'),
    [dash.dependencies.Input('full-time-graph', 'clickData'),
     dash.dependencies.Input('crossfilter-crypto-select', 'value'),
     dash.dependencies.Input('crossfilter-crypto-radio', 'value')])
def update_halfyear_by_click(click_data, bases, regression_base):
    """ Displays candlestick charts of the selected time and max time aggs back
        and a one dimensional data regression df with the same Datetimeindex as the input df.
    """
    graph_bases = []
    aggregation = "H"
    end = pd.Timestamp.now(tz='UTC')
    if bases is not None and len(bases) > 0:
        if click_data is None:
            end = cdf[bases[0]].index[len(cdf[bases[0]])-1]
        else:
            end = pd.Timestamp(click_data['points'][0]['x'], tz='UTC')
    start = end - pd.Timedelta(365/2, "D")  # one year

    aggmin = (end-start)/pd.Timedelta(60, "m")  # aggregate hourly
    if regression_base is not None:
        dcdf = cdf[regression_base]
        dcdf = dcdf.loc[(dcdf.index >= start) & (dcdf.index <= end)]
        dcdf = dcdf.resample(aggregation).agg({"open": "first"})
        if (dcdf is not None) and (len(dcdf) > 0):
            normfactor = dcdf.iloc[0].open
            dcdf = dcdf.apply(lambda x: (x / normfactor - 1) * 100)  # normalize to % change
        delta, Y_pred = time_linear_regression(dcdf["open"])
        legendname = "{:4.0f} h = delta/h: {:4.3f}".format(aggmin, delta)
        graph_bases.append(
            dict(
                x=dcdf.index,
                y=Y_pred,
                mode='lines',  # 'lines+markers'
                name=legendname))

    if bases is None:
        bases = []

    timeinfo = aggregation + ": " + start.strftime(Env.dt_format) + " - " + end.strftime(Env.dt_format)
    for base in bases:
        # dcdf = cdf[base].resample(aggregation).agg({"open": "first"})
        dcdf = cdf[base]
        dcdf = dcdf.loc[(dcdf.index >= start) & (dcdf.index <= end)]
        dcdf = dcdf.resample(aggregation).agg({"open": "first"})
        if (dcdf is not None) and (len(dcdf) > 0):
            normfactor = dcdf.iloc[0].open
            dcdf = dcdf.apply(lambda x: (x / normfactor - 1) * 100)  # normalize to % change
        graph_bases.append(dict(
            x=dcdf.index,
            y=dcdf["open"],
            mode='lines',  # 'lines+markers'
            name=base
        ))
    return {
        'data': graph_bases,
        'layout': {
            'height': 450,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': "normalized crypto prices"
            }],
            'yaxis': {'type': 'linear'},
            'xaxis': {'showgrid': False, 'title': timeinfo}
        }
    }


@app.callback(
    dash.dependencies.Output('ten-day-time-graph', 'figure'),
    [dash.dependencies.Input('halfyear-graph', 'clickData'),
     dash.dependencies.Input('crossfilter-crypto-select', 'value'),
     dash.dependencies.Input('crossfilter-crypto-radio', 'value')])
def update_ten_day_by_click(click_data, bases, regression_base):
    """ Displays candlestick charts of the selected time and max time aggs back
        and a one dimensional data regression df with the same Datetimeindex as the input df.
    """
    graph_bases = []
    aggregation = "5T"
    end = pd.Timestamp.now(tz='UTC')
    if bases is not None and len(bases) > 0:
        if click_data is None:
            end = cdf[bases[0]].index[len(cdf[bases[0]])-1]
        else:
            end = pd.Timestamp(click_data['points'][0]['x'], tz='UTC')
    start = end - pd.Timedelta(Env.minimum_minute_df_len, "m")

    aggmin = (end-start)/pd.Timedelta(1, "m")
    if regression_base is not None:
        dcdf = cdf[regression_base]
        dcdf = dcdf.loc[(dcdf.index >= start) & (dcdf.index <= end)]
        dcdf = dcdf.resample(aggregation).agg({"open": "first", "close": "last", "high": "max",
                                               "low": "min", })  # "volume": "sum"
        if (dcdf is not None) and (len(dcdf) > 0):
            normfactor = dcdf.iloc[0].open
            dcdf = dcdf.apply(lambda x: (x / normfactor - 1) * 100)  # normalize to % change
        delta, Y_pred = time_linear_regression(dcdf["open"])
        legendname = "{:4.0f} min = delta/h: {:4.3f}".format(aggmin, delta)
        graph_bases.append(
            dict(
                x=dcdf.index,
                y=Y_pred,
                mode='lines',  # 'lines+markers'
                name=legendname))

    if bases is None:
        bases = []

    timeinfo = aggregation + ": " + start.strftime(Env.dt_format) + " - " + end.strftime(Env.dt_format)
    for base in bases:
        # dcdf = cdf[base].resample(aggregation).agg({"open": "first"})
        dcdf = cdf[base]
        dcdf = dcdf.loc[(dcdf.index >= start) & (dcdf.index <= end)]
        dcdf = dcdf.resample(aggregation).agg({"open": "first", "close": "last", "high": "max",
                                               "low": "min", })  # "volume": "sum"
        if (dcdf is not None) and (len(dcdf) > 0):
            normfactor = dcdf.iloc[0].open
            dcdf = dcdf.apply(lambda x: (x / normfactor - 1) * 100)  # normalize to % change
        graph_bases.append(dict(
            x=dcdf.index,
            y=dcdf["open"],
            mode='lines',  # 'lines+markers'
            name=base
        ))
    return {
        'data': graph_bases,
        'layout': {
            'height': 450,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': "normalized crypto prices"
            }],
            'yaxis': {'type': 'linear'},
            'xaxis': {'showgrid': False, 'title': timeinfo}
        }
    }


def time_linear_regression(df):
    """ Receives a one column data dataframe with a Datetimeindex in fixed frequency.

        Returns a (hourly_delta, regression_df) tuple containing the regression delta within 1 hour
        and a one dimensional data regression df with the same Datetimeindex as the input df.
    """
    X = np.arange(len(df)).reshape(-1, 1)
    Y = df.values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)[:, 0]  # make predictions
    df = df.to_frame()
    df = df.assign(Y_regression=Y_pred)
    df["delta"] = (df.Y_regression - df.Y_regression.shift(1))
    # print("df.Y_reg", df.head(5))
    delta = Y_pred[1]-Y_pred[0]
    timediff_factor = pd.Timedelta(60, unit="m") / (df.index[1]-df.index[0])
    delta = delta * timediff_factor
    return (delta, df.Y_regression)


@app.callback(
    dash.dependencies.Output('full-day-time-graph', 'figure'),
    [dash.dependencies.Input('ten-day-time-graph', 'clickData'),
     dash.dependencies.Input('crossfilter-crypto-select', 'value'),
     dash.dependencies.Input('crossfilter-crypto-radio', 'value')])
def update_full_day_by_click(click_data, bases, regression_base):
    """ Displays candlestick charts of the selected time and max time aggs back
        together with the 10 day regression line from the radioi selected base
    """
    graph_bases = []
    aggregation = "T"
    end = pd.Timestamp.now(tz='UTC')
    if bases is not None and len(bases) > 0:
        if click_data is None:
            end = cdf[bases[0]].index[len(cdf[bases[0]])-1]
        else:
            end = pd.Timestamp(click_data['points'][0]['x'], tz='UTC')
    start = end - pd.Timedelta(24*60, "m")

    aggmin = (end-start)/pd.Timedelta(1, "m")
    if regression_base is not None:
        dcdf = cdf[regression_base]
        dcdf = dcdf.loc[(dcdf.index >= start) & (dcdf.index <= end)]
        dcdf = dcdf.resample(aggregation).agg({"open": "first", "close": "last", "high": "max",
                                               "low": "min", })  # "volume": "sum"
        if (dcdf is not None) and (len(dcdf) > 0):
            normfactor = dcdf.iloc[0].open
            dcdf = dcdf.apply(lambda x: (x / normfactor - 1) * 100)  # normalize to % change
        delta, Y_pred = time_linear_regression(dcdf["open"])
        legendname = "{:4.0f} min = delta/h: {:4.3f}".format(aggmin, delta)
        graph_bases.append(
            dict(
                x=dcdf.index,
                y=Y_pred,
                mode='lines',  # 'lines+markers'
                name=legendname))

    if bases is None:
        bases = []

    timeinfo = aggregation + ": " + start.strftime(Env.dt_format) + " - " + end.strftime(Env.dt_format)
    for base in bases:
        dcdf = cdf[base]
        dcdf = dcdf.loc[(dcdf.index >= start) & (dcdf.index <= end)]
        dcdf = dcdf.resample(aggregation).agg({"open": "first", "close": "last", "high": "max",
                                               "low": "min", })  # "volume": "sum"
        if (dcdf is not None) and (len(dcdf) > 0):
            normfactor = dcdf.iloc[0].open
            dcdf = dcdf.apply(lambda x: (x / normfactor - 1) * 100)  # normalize to % change
        graph_bases.append(dict(
            x=dcdf.index,
            y=dcdf["open"],
            mode='lines',  # 'lines+markers'
            name=base
        ))

    return {
        'data': graph_bases,
        'layout': {
            'height': 450,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': "normalized crypto prices"
            }],
            'yaxis': {'type': 'linear'},
            'xaxis': {'showgrid': False, 'title': timeinfo}
        }
    }


def normalize_data(dcdf, start, end, aggregation="T"):
    dcdf = dcdf.loc[(dcdf.index >= start) & (dcdf.index <= end)]
    if aggregation != "T":
        dcdf = dcdf.resample(aggregation).agg({"open": "first", "close": "last", "high": "max",
                                               "low": "min", })  # "volume": "sum"
    if (dcdf is not None) and (len(dcdf) > 0):
        normfactor = dcdf.iloc[0].open
        dcdf = dcdf.apply(lambda x: (x / normfactor - 1) * 100)  # normalize to % change
    return dcdf


def regression_graph(start, end, dcdf):
    reduced_dcdf = dcdf.loc[(dcdf.index >= start) & (dcdf.index <= end)]
    delta, Y_pred = time_linear_regression(reduced_dcdf["open"])
    aggmin = (end - start) / pd.Timedelta(1, "m")
    legendname = "{:4.0f} min = delta/h: {:4.3f}".format(aggmin, delta)
    return dict(x=reduced_dcdf.index, y=Y_pred, mode='lines', name=legendname, yaxis='y')  # 'lines+markers'


def volume_graph(start, end, dcdf):
    dcdf = dcdf.loc[(dcdf.index >= start) & (dcdf.index <= end)]
    INCREASING_COLOR = '#17BECF'
    DECREASING_COLOR = '#7F7F7F'
    colors = []
    for i in range(len(dcdf.close)):
        if i != 0:
            if dcdf.close[i] > dcdf.close[i-1]:
                colors.append(INCREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        else:
            colors.append(DECREASING_COLOR)
    return dict(x=dcdf.index, y=dcdf.volume, marker=dict(color=colors), type='bar', yaxis='y2', name='Volume' )


@app.callback(
    dash.dependencies.Output('zoom-in-graph', 'figure'),
    [dash.dependencies.Input('full-day-time-graph', 'clickData'),
     dash.dependencies.Input('crossfilter-crypto-radio', 'value'),
     dash.dependencies.Input('crossfilter-indicator-select', 'value')])
def update_detail_graph_by_click(click_data, base, indicators):
    """ Displays candlestick charts of the selected time and max time aggs back
        together with selected indicators
    """
    graph_bases = []
    dcdf = cdf[base]

    aggregation = "T"
    end = pd.Timestamp.now(tz='UTC')
    if base is not None:
        if click_data is None:
            end = cdf[base].index[len(cdf[base])-1]
        else:
            end = pd.Timestamp(click_data['points'][0]['x'], tz='UTC')
    start = end - pd.Timedelta(4*60, "m")
    ndcdf = normalize_data(dcdf, start, end, aggregation)
    graph_bases.append(
        go.Candlestick(x=ndcdf.index, open=ndcdf.open, high=ndcdf.high, low=ndcdf.low,
                       close=ndcdf.close, yaxis='y'))
    graph_bases.append(volume_graph(start, end, dcdf))

    if indicators is None:
        indicators = []

    graph_bases.append(regression_graph(start, end, ndcdf))
    graph_bases.append(regression_graph(end - pd.Timedelta(5, "m"), end, ndcdf))
    graph_bases.append(regression_graph(end - pd.Timedelta(11, "m"), end - pd.Timedelta(6, "m"), ndcdf))
    graph_bases.append(regression_graph(end - pd.Timedelta(30, "m"), end, ndcdf))

    timeinfo = aggregation + ": " + start.strftime(Env.dt_format) + " - " + end.strftime(Env.dt_format)
    return {
        'data': graph_bases,
        'layout': {
            'height': 650,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': "normalized crypto prices"
            }],
            'yaxis': {'type': 'linear', "domain": [0.2, 0.8]},
            'yaxis2': {"domain": [0., 0.2], "showticklabels": False},
            'xaxis': {'showgrid': False, 'title': timeinfo}
        }
    }


# @app.callback(
#     dash.dependencies.Output('volume-signals-graph', 'figure'),
#     [dash.dependencies.Input('full-day-time-graph', 'clickData'),
#      dash.dependencies.Input('crossfilter-crypto-radio', 'value'),
#      dash.dependencies.Input('crossfilter-indicator-select', 'value')])
# def update_volume_signals__graph_by_click(click_data, base, indicators):
#     """ Displays volume and trade signals of the selected time and 4h back
#         together with selected indicators
#     """
#     graph_bases = []
#     dcdf = cdf[base]
#     INCREASING_COLOR = '#17BECF'
#     DECREASING_COLOR = '#7F7F7F'
#     colors = []

#     for i in range(len(dcdf.close)):
#         if i != 0:
#             if dcdf.close[i] > dcdf.close[i-1]:
#                 colors.append(INCREASING_COLOR)
#             else:
#                 colors.append(DECREASING_COLOR)
#         else:
#             colors.append(DECREASING_COLOR)

#     end = pd.Timestamp.now(tz='UTC')
#     if base is not None:
#         # print(base)
#         if click_data is None:
#             end = cdf[base].index[len(cdf[base])-1]
#         else:
#             end = pd.Timestamp(click_data['points'][0]['x'], tz='UTC')
#     start = end - pd.Timedelta(4*60, "m")
#     aggregation = "T"
#     timeinfo = aggregation + ": " + start.strftime(Env.dt_format) + " - " + end.strftime(Env.dt_format)
#     dcdf = dcdf.loc[(dcdf.index >= start) & (dcdf.index <= end)]
#     if (dcdf is not None) and (len(dcdf) > 0):
#         normfactor = dcdf.iloc[0].open
#         dcdf = dcdf.apply(lambda x: (x / normfactor - 1) * 100)  # normalize to % change
#     # dcdf = dcdf.resample(aggregation).agg({"open": "first", "close": "last", "high": "max",
#     #                                        "low": "min", })  # "volume": "sum"
#     # print("zoom-in", bases, indicators, start, end)
#     go.Candlestick(x=dcdf.index, open=dcdf.open, high=dcdf.high, low=dcdf.low, close=dcdf.close)

#     # dcc.Graph(
#     #         id='example-graph',
#     #         figure={
#     #             'data': [
#     #                 {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
#     #                 {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
#     #             ],
#     #             'layout': {
#     #                 'title': 'Dash Data Visualization'
#     #             }
#     #         }
#     #     )
#     return {
#         'data': graph_bases,
#         'layout': {
#             'height': 450,
#             'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
#             'annotations': [{
#                 'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
#                 'xref': 'paper', 'yref': 'paper', 'showarrow': False,
#                 'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
#                 'text': "normalized crypto prices"
#             }],
#             'yaxis': {'type': 'linear'},
#             'xaxis': {'showgrid': False, 'title': timeinfo}
#         }
#     }


def load_crypto_data():
    # cdf = {base: ccd.load_asset_dataframe(base) for base in Env.usage.bases}
    [print(base, len(cdf[base])) for base in cdf]
    print("xrp minute  aggregated", cdf["xrp"].head())
    dcdf = cdf["xrp"].resample("D").agg({"open": "first", "close": "last", "high": "max",
                                         "low": "min", "volume": "sum"})
    print("xrp day aggregated", dcdf.head())


if __name__ == '__main__':
    # load_crypto_data()
    cdf = {base: ccd.load_asset_dataframe(base) for base in Env.usage.bases}
    app.run_server(debug=True)
