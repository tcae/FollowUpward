import json
from textwrap import dedent

import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# import plotly.graph_objs as go
import pandas as pd
from env_config import Env
# import env_config as env
import crypto_targets as ct
import crypto_features as cf
import cached_crypto_data as ccd
import indicators as ind

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


def df_check(df, timerange):
    if timerange is None:
        print(f"timerange: {timerange}")
    else:
        print(f"timerange: all")
    print(df.head(1))
    print(df.tail(1))


@app.callback(
    dash.dependencies.Output('full-time-graph', 'figure'),
    [dash.dependencies.Input('crossfilter-crypto-select', 'value')])
def update_graph(bases):
    graph_bases = []
    if bases is not None:
        for base in bases:
            # df_check(cdf[base], None)
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


def normalize_open(dcdf, start, end, aggregation):
    dcdf = dcdf.loc[(dcdf.index >= start) & (dcdf.index <= end)]
    if aggregation != "T":
        dcdf = dcdf.resample(aggregation).agg({"open": "first"})
    if (dcdf is not None) and (len(dcdf) > 0):
        normfactor = dcdf.iloc[0].open
        dcdf = dcdf.apply(lambda x: (x / normfactor - 1) * 100)  # normalize to % change
    return dcdf


def normalize_ohlc(dcdf, start, end, aggregation):
    dcdf = dcdf.loc[(dcdf.index >= start) & (dcdf.index <= end)]
    if aggregation != "T":
        dcdf = dcdf.resample(aggregation).agg({"open": "first", "close": "last", "high": "max",
                                               "low": "min"})  # "volume": "sum"
    if (dcdf is not None) and (len(dcdf) > 0):
        normfactor = dcdf.iloc[0].open
        dcdf = dcdf.apply(lambda x: (x / normfactor - 1) * 100)  # normalize to % change
    return dcdf


def regression_graph(start, end, dcdf, aggregation):
    reduced_dcdf = dcdf.loc[(dcdf.index >= start) & (dcdf.index <= end)]
    delta, Y_pred = ind.time_linear_regression(reduced_dcdf["open"])
    aggmin = (end - start) / pd.Timedelta(1, aggregation)
    legendname = "{:4.0f} {} = delta/h: {:4.3f}".format(aggmin, aggregation, delta)
    return dict(x=reduced_dcdf.index[[0, -1]], y=Y_pred, mode='lines', name=legendname, yaxis='y')


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
    return dict(x=dcdf.index, y=dcdf.volume, marker=dict(color=colors), type='bar', yaxis='y2', name='Volume')


def open_timeline_graph(timerange, aggregation, click_data, bases, regression_base, indicators):
    """ Displays a line chart of open prices
        and a one dimensional data regression df with the same Datetimeindex as the input df.
    """
    graph_bases = []
    end = pd.Timestamp.now(tz='UTC')
    if bases is not None and len(bases) > 0:
        if click_data is None:
            end = cdf[bases[0]].index[len(cdf[bases[0]])-1]
        else:
            end = pd.Timestamp(click_data['points'][0]['x'], tz='UTC')
    start = end - timerange

    if indicators is None:
        indicators = []
    else:
        if "regression 1D" in indicators:
            if regression_base is not None:
                reduced_dcdf = normalize_open(cdf[regression_base], start, end, aggregation)
                graph_bases.append(regression_graph(start, end, reduced_dcdf, aggregation))

    if bases is None:
        bases = []
    for base in bases:
        # df_check(cdf[base], timerange)
        dcdf = normalize_open(cdf[base], start, end, aggregation)
        graph_bases.append(dict(x=dcdf.index, y=dcdf["open"],  mode='lines', name=base))

    timeinfo = aggregation + ": " + start.strftime(Env.dt_format) + " - " + end.strftime(Env.dt_format)
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
    dash.dependencies.Output('halfyear-graph', 'figure'),
    [dash.dependencies.Input('full-time-graph', 'clickData'),
     dash.dependencies.Input('crossfilter-crypto-select', 'value'),
     dash.dependencies.Input('crossfilter-crypto-radio', 'value'),
     dash.dependencies.Input('crossfilter-indicator-select', 'value')])
def update_halfyear_by_click(click_data, bases, regression_base, indicators):
    aggregation = "H"
    timerange = pd.Timedelta(365/2, "D")
    return open_timeline_graph(timerange, aggregation, click_data, bases, regression_base, indicators)


@app.callback(
    dash.dependencies.Output('ten-day-time-graph', 'figure'),
    [dash.dependencies.Input('halfyear-graph', 'clickData'),
     dash.dependencies.Input('crossfilter-crypto-select', 'value'),
     dash.dependencies.Input('crossfilter-crypto-radio', 'value'),
     dash.dependencies.Input('crossfilter-indicator-select', 'value')])
def update_ten_day_by_click(click_data, bases, regression_base, indicators):
    aggregation = "T"
    timerange = pd.Timedelta(Env.minimum_minute_df_len, "T")
    return open_timeline_graph(timerange, aggregation, click_data, bases, regression_base, indicators)


@app.callback(
    dash.dependencies.Output('full-day-time-graph', 'figure'),
    [dash.dependencies.Input('ten-day-time-graph', 'clickData'),
     dash.dependencies.Input('crossfilter-crypto-select', 'value'),
     dash.dependencies.Input('crossfilter-crypto-radio', 'value'),
     dash.dependencies.Input('crossfilter-indicator-select', 'value')])
def update_full_day_by_click(click_data, bases, regression_base, indicators):
    aggregation = "T"
    timerange = pd.Timedelta(24*60, "T")
    return open_timeline_graph(timerange, aggregation, click_data, bases, regression_base, indicators)


def target_list(base, start, end, dcdf):
    # labels1 = [i for i in range(241)]
    # print(len(labels1), labels1)
    # return labels1, None
    target_dict = dict()
    # fstart = start - pd.Timedelta(Env.minimum_minute_df_len, "m")
    fstart = start - pd.Timedelta(Env.minimum_minute_df_len, "m")
    fdf = dcdf.loc[(dcdf.index >= start) & (dcdf.index <= end)]
    tf = cf.TargetsFeatures(base, minute_dataframe=fdf)
    dcdf = tf.minute_data.loc[(tf.minute_data.index >= start) & (tf.minute_data.index <= end)]

    # targets = [t for t in dcdf["target_thresholds"]]
    # labels = [ct.TARGET_NAMES[t] for t in targets]
    # target_dict["target_thresholds"] = {"targets": targets, "labels": labels}

    targets = [t for t in dcdf["target2"]]
    labels = [ct.TARGET_NAMES[t] for t in targets]
    target_dict["newtargets"] = {"targets": targets, "labels": labels}

    targets = [t for t in dcdf["target"]]
    labels = [ct.TARGET_NAMES[t] for t in targets]
    target_dict["target1"] = {"targets": targets, "labels": labels}

    # labels2 = [ct.TARGET_NAMES[t] for t in dcdf["target2"]]
    # print(len(labels1), labels1, len(targets1), targets1)
    # print(len(labels2), labels2)
    return target_dict


def target_heatmap(base, start, end, dcdf, target_dict):
    # dcdf neds to be expanded to get more history for feature calculation
    dcdf = dcdf.loc[(dcdf.index >= start) & (dcdf.index <= end)]
    # df_check(dcdf, pd.Timedelta(end-start, "m"))
    return go.Heatmap(
        x=dcdf.index, y=[i for i in target_dict],
        z=[target_dict[t]["targets"] for t in target_dict], zmin=-1, zmax=1,
        yaxis='y3', name='labels',
        text=[target_dict[t]["labels"] for t in target_dict],
        colorscale='RdYlGn',
        reversescale=False,
        hoverinfo="x+y+z+text+name",
        showscale=False,
        # autocolorscale=False,
        # colorscale=[[cf.TARGETS[cf.HOLD]/2, "rgb(255, 234, 0)"],
        #             [cf.TARGETS[cf.SELL]/2, "rgb(255, 0, 0)"],
        #             [cf.TARGETS[cf.BUY]/2, "rgb(116, 196, 118)"]],
        colorbar=dict(tick0=0, dtick=1))


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
    zoom_in_time = 4*60

    aggregation = "T"
    if indicators is None:
        indicators = []
    end = pd.Timestamp.now(tz='UTC')
    if click_data is None:
        end = cdf[base].index[len(cdf[base])-1]
    else:
        end = pd.Timestamp(click_data['points'][0]['x'], tz='UTC')
    start = end - pd.Timedelta(zoom_in_time, "m")

    ndcdf = normalize_ohlc(dcdf, start, end, aggregation)
    # print(f"update_detail_graph_by_click: len(ndcdf): {len(ndcdf)}, len(labels): {len(labels1)}")
    if "targets" in indicators:
        target_dict = target_list(base, start, end, dcdf)
        graph_bases.append(
            go.Candlestick(x=ndcdf.index, open=ndcdf.open, high=ndcdf.high, low=ndcdf.low,
                           close=ndcdf.close, yaxis='y',
                           hoverinfo="x+y+z+text+name", text=target_dict["target1"]["labels"]))
        graph_bases.append(target_heatmap(base, start, end, ndcdf, target_dict))
    else:
        graph_bases.append(
            go.Candlestick(x=ndcdf.index, open=ndcdf.open, high=ndcdf.high, low=ndcdf.low,
                           close=ndcdf.close, yaxis='y',
                           hoverinfo="x+y+z+text+name"))

    if "regression 1D" in indicators:
        graph_bases.append(regression_graph(start, end, ndcdf, aggregation))
        graph_bases.append(regression_graph(end - pd.Timedelta(5, "m"), end, ndcdf, aggregation))
        graph_bases.append(regression_graph(end - pd.Timedelta(11, "m"),
                           end - pd.Timedelta(6, "m"), ndcdf, aggregation))
        graph_bases.append(regression_graph(end - pd.Timedelta(30, "m"), end, ndcdf, aggregation))
    graph_bases.append(volume_graph(start, end, dcdf))

    timeinfo = aggregation + ": " + start.strftime(Env.dt_format) + " - " + end.strftime(Env.dt_format)
    return {
        'data': graph_bases,
        'layout': {
            'height': 900,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': "normalized crypto prices"
            }],
            'yaxis2': {"domain": [0., 0.2]},
            'yaxis3': {"domain": [0.2, 0.23], "showticklabels": False},
            'yaxis': {'type': 'linear', "domain": [0.3, 1]},
            'xaxis': {'showgrid': False, 'title': timeinfo, "rangeslider": {"visible": False}}
        }
    }


def load_crypto_data():
    # cdf = {base: ccd.load_asset_dataframe(base) for base in Env.usage.bases}
    [print(base, len(cdf[base])) for base in cdf]
    print("xrp minute  aggregated", cdf["xrp"].head())
    dcdf = cdf["xrp"].resample("D").agg({"open": "first", "close": "last", "high": "max",
                                         "low": "min", "volume": "sum"})
    print("xrp day aggregated", dcdf.head())


if __name__ == '__main__':
    # load_crypto_data()
    # env.test_mode()
    cdf = {base: ccd.load_asset_dataframe(base, path=Env.data_path) for base in Env.usage.bases}
    app.run_server(debug=True)
