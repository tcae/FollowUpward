import json
import logging
from textwrap import dedent

import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# import plotly.graph_objs as go
import pandas as pd
from env_config import Env
import env_config as env
import crypto_targets as ct
import condensed_features as cof
import cached_crypto_data as ccd
import indicators as ind
import crypto_history_sets as chs

logger = logging.getLogger(__name__)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
ohlcv_df_dict = None  # ohlcv minute data
features = None
targets = None

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}
indicator_opts = ["regression 1D", "targets", "features"]


app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Checklist(
                id='crossfilter-crypto-select',
                options=[{'label': i, 'value': i} for i in Env.bases],
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
            options=[{'label': i, 'value': i} for i in Env.bases],
            labelStyle={'display': 'inline-block'}
        ),
        dcc.Checklist(
            id='crossfilter-indicator-select',
            options=[{'label': i, 'value': i} for i in indicator_opts],
            value=[],
            labelStyle={'display': 'inline-block'}
        ),
        dcc.Graph(id='zoom-in-graph'),
        # dcc.Graph(id='volume-signals-graph'),
    ], style={'display': 'inline-block', 'float': 'right', 'width': '49%'}),

])


def df_check(df, timerange):
    if timerange is not None:
        logger.debug(f"timerange: {timerange} len(df): {len(df)}")
    else:
        logger.debug(f"timerange: all len(df): {len(df)}")
    logger.debug(str(df.head(1)))
    logger.debug(str(df.tail(1)))


@app.callback(
    dash.dependencies.Output('full-time-graph', 'figure'),
    [dash.dependencies.Input('crossfilter-crypto-select', 'value')])
def update_graph(bases):
    graph_bases = []
    if bases is not None:
        for base in bases:
            # df_check(ohlcv_df_dict[base], None)
            bmd = ohlcv_df_dict[base].resample("D").agg({"close": "first"})  # bmd == base minute data
            # bmd = ohlcv_df_dict[base].resample("D").agg({"open": "first", "close": "last", "high": "max",
            #                                     "low": "min", "volume": "sum"})
            bmd = bmd/bmd.max()  # normalize
            graph_bases.append(dict(
                x=bmd.index,
                y=bmd["close"],
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
            return Env.bases
        else:
            return []
    if all_click_ts is not None:
        return Env.bases
    if none_click_ts is not None:
        return []
    return [Env.bases[0]]


@app.callback(
    Output('crossfilter-crypto-radio', 'value'),
    [Input('crossfilter-crypto-select', 'value')])
def radio_range(selected_cryptos):
    if (selected_cryptos is not None) and (len(selected_cryptos) > 0):
        return selected_cryptos[0]
    else:
        return Env.bases[0]


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


def normalize_close(bmd, start, end, aggregation):
    bmd = bmd.loc[(bmd.index >= start) & (bmd.index <= end)]
    if aggregation != "T":
        bmd = bmd.resample(aggregation).agg({"close": "first"})
    if (bmd is not None) and (len(bmd) > 0):
        normfactor = bmd.iloc[0].close
        bmd = bmd.apply(lambda x: (x / normfactor - 1) * 100)  # normalize to % change
    return bmd


def normalize_ohlc(bmd, start, end, aggregation):
    bmd = bmd.loc[(bmd.index >= start) & (bmd.index <= end)]
    if aggregation != "T":
        bmd = bmd.resample(aggregation).agg({"open": "first", "close": "last", "high": "max",
                                             "low": "min"})  # "volume": "sum"
    if (bmd is not None) and (len(bmd) > 0):
        normfactor = bmd.iloc[0].close
        bmd = bmd.apply(lambda x: (x / normfactor - 1) * 100)  # normalize to % change
    return bmd


def volume_graph(start, end, bmd):
    bmd = bmd.loc[(bmd.index >= start) & (bmd.index <= end)]
    INCREASING_COLOR = '#17BECF'
    DECREASING_COLOR = '#7F7F7F'
    colors = []
    for i in range(len(bmd.close)):
        if i != 0:
            if bmd.close[i] > bmd.close[i-1]:
                colors.append(INCREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        else:
            colors.append(DECREASING_COLOR)
    return dict(x=bmd.index, y=bmd.volume, marker=dict(color=colors), type='bar', yaxis='y2', name='Volume')


def straigt_line(start, end, end_y, gain_per_hour, legendname):
    """ Input: an 'end' timestamp with the corresponding y value 'end_y' and the gain per hour.

        Returns the plotly straight line graph between 'start' and 'end'.
    """
    start_y = end_y - float((end - start) / pd.Timedelta(60, "T")) * gain_per_hour
    return dict(x=[start, end], y=[start_y, end_y], mode="lines", name=legendname, yaxis='y')


def chance_risk_marker(start, end, end_y, chance, risk, legendname):
    """ Input: an 'end' timestamp with the corresponding y value 'end_y' and the chance and risk.

        Returns the plotly straight marker graph at 'end'.
    """
    return dict(x=[end, end], y=[end_y+chance, end_y-risk], mode="marker", name=legendname, yaxis='y')


def volume_ratio_marker(time, vol_ratio, legendname):
    """ Input: a 'time' timestamp with the corresponding y value 'vol_rel'.

        Returns the plotly straight marker graph.
    """
    return dict(x=[time], y=[vol_ratio], mode="scatter", name=legendname, yaxis='y')


def show_condensed_features(base, start, time):
    """ Input 'time' timestamp of period and base minute data 'bmd' for which features are required.
        'bmd' shall have history data of cof.MHE minutes before 'time' for feature calculation.

        Returns a plotly graph that visualizes the features at timestamp 'time'. The graph starts at 'start'.
    """
    regr_start = time - pd.Timedelta(features.history(), "T")
    bmd = ohlcv_df_dict[base]
    reduced_bmd = bmd.loc[(bmd.index >= regr_start) & (bmd.index <= time)]

    normfactor = bmd.loc[start, "close"]
    # df_check(reduced_bmd, pd.Timedelta(time-start, "m"))
    red_bmdc = reduced_bmd.apply(lambda x: (x / normfactor - 1) * 100, result_type="broadcast")  # normalize to % change
    red_bmdc.loc[:, "volume"] = reduced_bmd["volume"]
    # df_check(red_bmdc, pd.Timedelta(time-start, "m"))

    bf_df = cof.cal_features(red_bmdc)  # bf_df == base features
    # bf_df = cof.calc_features_nocache(red_bmdc)  # bf_df == base features
    # df_check(bf_df, pd.Timedelta(time-start, "m"))

    start = max(start, regr_start)
    for (regr_only, offset, minutes, ext) in cof.REGRESSION_KPI:
        time_regr_y = float(bf_df["price"+ext])
        gain = float(bf_df["gain"+ext])
        line_end = time - pd.Timedelta(offset, "T")
        line_start = line_end - pd.Timedelta(minutes-1, "T")
        line_start = max(line_start, start)
        legendname = "{}: d/h = {:4.3f}".format(ext, gain)
        yield straigt_line(line_start, line_end, time_regr_y, gain, legendname)
        if not regr_only:
            time_cur_y = float(red_bmdc.loc[line_end, "close"])
            chance = float(bf_df["chance"+ext])
            risk = float(bf_df["risk"+ext])
            legendname = "{}: chance/risk = {:4.3f}/{:4.3f}".format(ext, chance, risk)
            yield chance_risk_marker(line_start, line_end, time_cur_y, chance, risk, legendname)
    for (svol, lvol, ext) in cof.VOL_KPI:
        # df["vol"+ext][tic] = __vol_rel(volumes, lvol, svol)
        legendname = "vol"+ext
        vol_ratio = float(bf_df.loc[time, "vol"+ext])
        yield volume_ratio_marker(time, vol_ratio, legendname)

    return None


def regression_graph(start, end, bmd, aggregation):
    reduced_bmd = bmd.loc[(bmd.index >= start) & (bmd.index <= end)]
    delta, Y_pred = ind.time_linear_regression(reduced_bmd["close"])
    aggmin = (end - start) / pd.Timedelta(1, aggregation)
    legendname = "{:4.0f} {} = delta/h: {:4.3f}".format(aggmin, aggregation, delta)
    return dict(x=reduced_bmd.index[[0, -1]], y=Y_pred, mode='lines', name=legendname, yaxis='y')


def close_timeline_graph(timerange, aggregation, click_data, bases, regression_base, indicators):
    """ Displays a line chart of close prices
        and a one dimensional close prices regression line both on the same Datetimeindex.
    """
    graph_bases = []
    end = pd.Timestamp.now(tz='UTC')
    if bases is not None and len(bases) > 0:
        if click_data is None:
            end = ohlcv_df_dict[bases[0]].index[-1]
        else:
            end = pd.Timestamp(click_data['points'][0]['x'], tz='UTC')
    start = end - timerange

    if bases is None:
        bases = []
    for base in bases:
        reduced_bmd = normalize_close(ohlcv_df_dict[base], start, end, aggregation)
        regr = regression_graph(start, end, reduced_bmd, aggregation)
        if indicators is None:
            indicators = []
        else:
            if "regression 1D" in indicators:
                if regression_base == base:
                    graph_bases.append(regr)

        # df_check(ohlcv_df_dict[base], timerange)
        bmd = normalize_close(ohlcv_df_dict[base], start, end, aggregation)
        legendname = base + " " + regr["name"]
        graph_bases.append(dict(x=bmd.index, y=bmd["close"],  mode='lines', name=legendname))

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
    return close_timeline_graph(timerange, aggregation, click_data, bases, regression_base, indicators)


@app.callback(
    dash.dependencies.Output('ten-day-time-graph', 'figure'),
    [dash.dependencies.Input('halfyear-graph', 'clickData'),
     dash.dependencies.Input('crossfilter-crypto-select', 'value'),
     dash.dependencies.Input('crossfilter-crypto-radio', 'value'),
     dash.dependencies.Input('crossfilter-indicator-select', 'value')])
def update_ten_day_by_click(click_data, bases, regression_base, indicators):
    aggregation = "T"
    timerange = pd.Timedelta(chs.ActiveFeatures.history(), "T")
    return close_timeline_graph(timerange, aggregation, click_data, bases, regression_base, indicators)


@app.callback(
    dash.dependencies.Output('full-day-time-graph', 'figure'),
    [dash.dependencies.Input('ten-day-time-graph', 'clickData'),
     dash.dependencies.Input('crossfilter-crypto-select', 'value'),
     dash.dependencies.Input('crossfilter-crypto-radio', 'value'),
     dash.dependencies.Input('crossfilter-indicator-select', 'value')])
def update_full_day_by_click(click_data, bases, regression_base, indicators):
    aggregation = "T"
    timerange = pd.Timedelta(24*60, "T")
    return close_timeline_graph(timerange, aggregation, click_data, bases, regression_base, indicators)


def target_list(base, start, end, bmd):
    # labels1 = [i for i in range(241)]
    # logger.debug(f"{len(labels1)}, {labels1}")
    # return labels1, None
    colormap = {ct.TARGETS[ct.HOLD]: 0, ct.TARGETS[ct.BUY]: 1, ct.TARGETS[ct.SELL]: -1}

    target_dict = dict()
    target_df = targets.get_data(base, start, end)
    # fstart = start - pd.Timedelta(features.history(), "T")
    # fdf = bmd.loc[(bmd.index >= fstart) & (bmd.index <= end)]
    # tf = chs.ActiveFeatures(base, minute_dataframe=fdf)
    # tf.crypto_targets()
    # bmd = tf.minute_data.loc[(tf.minute_data.index >= start) & (tf.minute_data.index <= end)]

    # targets = [t for t in bmd["target_thresholds"]]
    # labels = [ct.TARGET_NAMES[t] for t in targets]
    # target_dict["target_thresholds"] = {"targets": targets, "labels": labels}

    # targets = [t for t in bmd["target2"]]
    # labels = [ct.TARGET_NAMES[t] for t in targets]
    # target_dict["newtargets"] = {"targets": targets, "labels": labels}

    target_colors = [colormap[t] for t in target_df["target"]]
    labels = [ct.TARGET_NAMES[t] for t in target_df["target"]]
    target_dict["target1"] = {"targets": target_colors, "labels": labels}

    # labels2 = [ct.TARGET_NAMES[t] for t in bmd["target2"]]
    # logger.debug(f"{len(labels1)}, {labels1}, {len(targets1)}, {targets1}")
    # logger.debug(f"{len(labels2)}, {labels2}")
    return target_dict


def target_heatmap(base, start, end, bmd, target_dict):
    # bmd neds to be expanded to get more history for feature calculation
    bmd = bmd.loc[(bmd.index >= start) & (bmd.index <= end)]
    # df_check(bmd, pd.Timedelta(end-start, "m"))
    return go.Heatmap(
        x=bmd.index, y=[i for i in target_dict],
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
    [dash.dependencies.Input('zoom-in-graph', 'clickData'),
     dash.dependencies.Input('full-day-time-graph', 'clickData'),
     dash.dependencies.Input('crossfilter-crypto-radio', 'value'),
     dash.dependencies.Input('crossfilter-indicator-select', 'value')])
def update_detail_graph_by_click(zoom_click, day_click, base, indicators):
    """ Displays candlestick charts of the selected time and max time aggs back
        together with selected indicators
    """
    graph_bases = []
    bmd = ohlcv_df_dict[base]
    zoom_in_time = 4*60

    aggregation = "T"
    if indicators is None:
        indicators = []
    end = pd.Timestamp.now(tz='UTC')
    if day_click is None:
        end = ohlcv_df_dict[base].index[-1]
    else:
        end = pd.Timestamp(day_click['points'][0]['x'], tz='UTC')
    start = end - pd.Timedelta(zoom_in_time, "m")

    nbmd = normalize_ohlc(bmd, start, end, aggregation)
    # logger.debug(f"update_detail_graph_by_click: len(nbmd): {len(nbmd)}, len(labels): {len(labels1)}")
    if "targets" in indicators:
        target_dict = target_list(base, start, end, bmd)
        graph_bases.append(
            go.Candlestick(x=nbmd.index, open=nbmd.open, high=nbmd.high, low=nbmd.low,
                           close=nbmd.close, yaxis='y',
                           hoverinfo="x+y+z+text+name", text=target_dict["target1"]["labels"]))
        graph_bases.append(target_heatmap(base, start, end, nbmd, target_dict))
    else:
        graph_bases.append(
            go.Candlestick(x=nbmd.index, open=nbmd.open, high=nbmd.high, low=nbmd.low,
                           close=nbmd.close, yaxis='y',
                           hoverinfo="x+y+z+text+name"))

    if "regression 1D" in indicators:
        graph_bases.append(regression_graph(start, end, nbmd, aggregation))
        graph_bases.append(regression_graph(end - pd.Timedelta(5-1, "m"), end, nbmd, aggregation))
        graph_bases.append(regression_graph(end - pd.Timedelta(10-1, "m"),
                           end - pd.Timedelta(5, "m"), nbmd, aggregation))
        graph_bases.append(regression_graph(end - pd.Timedelta(30-1, "m"), end, nbmd, aggregation))
    graph_bases.append(volume_graph(start, end, bmd))

    if ("features" in indicators) and (zoom_click is not None):
        clicked_time = pd.Timestamp(zoom_click['points'][0]['x'], tz='UTC')
        for graph in show_condensed_features(base, start, clicked_time):
            # logger.debug(str(graph))
            graph_bases.append(graph)

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
            'yaxis2': {"domain": [0., 0.2], "side": "left"},
            'yaxis4': {"domain": [0., 0.2], "side": "right", "overlaying": "y"},
            'yaxis3': {"domain": [0.2, 0.23], "showticklabels": False},
            'yaxis': {'type': 'linear', "domain": [0.3, 1]},
            'xaxis': {'showgrid': False, 'title': timeinfo, "rangeslider": {"visible": False}}
        }
    }


if __name__ == '__main__':
    # load_crypto_data()
    # env.test_mode()
    tee = env.Tee(log_prefix="TrainEval")
    ohlcv = ccd.Ohlcv()
    features = cof.F2cond20(ohlcv)
    targets = ct.T10up5low30min(ohlcv)
    ohlcv_df_dict = {base: ohlcv.load_data(base) for base in Env.bases}
    app.run_server(debug=True)
