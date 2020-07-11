import json
import logging
from textwrap import dedent

import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
# import dash_table.FormatTemplate as FormatTemplate
# from dash_table.Format import Sign
from dash_table.Format import Format, Scheme
# import plotly.graph_objs as go
import pandas as pd
from env_config import Env
import env_config as env
import crypto_targets as ct
import condensed_features as cof
import cached_crypto_data as ccd
import indicators as ind
import update_crypto_history as uch
# import classifier_predictor as cp

logger = logging.getLogger(__name__)

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
ohlcv_df_dict = None  # ohlcv minute data
features = None
targets = None
classifier_set = None

styles = {
    "pre": {
        "border": "thin lightgrey solid",
        "overflowX": "scroll"
    }
}
indicator_opts = ["regression 1D", "targets", "signals", "features", "equal scale"]
view_config = {
    "graph4h": {
        "legend": "4h", "timerange": pd.Timedelta(4, "h"), "aggregation": "T",
        "regression": {
            "4h": {"timerange": pd.Timedelta(4, "h"), "aggregation": "m"},
            "2h": {"timerange": pd.Timedelta(2, "h"), "aggregation": "m"},
            "1h": {"timerange": pd.Timedelta(1, "h"), "aggregation": "m"},
            "30m": {"timerange": pd.Timedelta(30, "m"), "aggregation": "m"},
            "15m": {"timerange": pd.Timedelta(15, "m"), "aggregation": "m"},
            "5m": {"timerange": pd.Timedelta(5, "m"), "aggregation": "m"}
        }
    },
    "graph1day": {
        "legend": "1d", "timerange": pd.Timedelta(1, "D"), "aggregation": "T", "focusrange": pd.Timedelta(4, "h")},
    "graph10day": {
        "legend": "10d", "timerange": pd.Timedelta(10, "D"), "aggregation": "T", "focusrange": pd.Timedelta(1, "D")},
    "graph6month": {
        "legend": "6M", "timerange": pd.Timedelta(365/2, "D"), "aggregation": "h", "focusrange": pd.Timedelta(10, "D")},
    "graph_all": {"focusrange": pd.Timedelta(365/2, "D")},
    "kpi_table": {
        "regression": {
            "1m": {"timerange": pd.Timedelta(1, "m"), "aggregation": "m", "buy": 60., "sell": -30.},
            "5m": {"timerange": pd.Timedelta(5, "m"), "aggregation": "m", "buy": 12., "sell": -6.},
            "15m": {"timerange": pd.Timedelta(15, "m"), "aggregation": "m", "buy": 4., "sell": -2.},
            "30m": {"timerange": pd.Timedelta(30, "m"), "aggregation": "m", "buy": 2., "sell": -1.},
            "1h": {"timerange": pd.Timedelta(1, "h"), "aggregation": "m", "buy": 1., "sell": -0.5},
            "2h": {"timerange": pd.Timedelta(2, "h"), "aggregation": "m", "buy": 0.5, "sell": -0.25},
            "4h": {"timerange": pd.Timedelta(4, "h"), "aggregation": "m", "buy": 0.25, "sell": -0.125},
            "12h": {"timerange": pd.Timedelta(12, "h"), "aggregation": "m", "buy": 0.125, "sell": -0.0625},
        },
        "volume": {
            # "5m1h": {"shortrange": pd.Timedelta(5, "m"), "longrange": pd.Timedelta(1, "h")},
            "5m4h": {"shortrange": pd.Timedelta(5, "m"), "longrange": pd.Timedelta(4, "h")},
        },
        "liquidity": {
            # "5m1h": {"shortrange": pd.Timedelta(5, "m"), "longrange": pd.Timedelta(1, "h")},
            "median(1h)/min": {"timerange": pd.Timedelta(1, "h")},
        }
    },
}


app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Checklist(
                id="crypto_select",
                options=[{"label": i, "value": i} for i in Env.bases],
                labelStyle={"display": "inline-block"}
            ),
            html.Button("all", id="all_button", style={"display": "inline-block"}),
            html.Button("none", id="none_button", style={"display": "inline-block"}),
            html.Button("update data", id="update_data", style={"display": "inline-block"}),
        ], style={"display": "inline-block"}),
        # html.H1("Crypto Price"),  # style={"textAlign": "center"},
        dcc.Graph(id="graph1day"),
        dcc.Graph(id="graph10day"),
        dcc.Graph(id="graph6month"),
        dcc.Graph(id="graph_all"),
        html.Div(id="focus", style={"display": "none"}),
        html.Div(id="graph1day_end", style={"display": "none"}),
        html.Div(id="graph10day_end", style={"display": "none"}),
        html.Div(id="graph6month_end", style={"display": "none"}),
        html.Div(className="row", children=[
            html.Div([
                dcc.Markdown(dedent("""
                    **Hover Data**

                    Mouse over values in the graph.
                """)),
                html.Pre(id="hover-data", style=styles["pre"])
            ], className="three columns"),

            html.Div([
                dcc.Markdown(dedent("""
                    **Click Data**

                    Click on points in the graph.
                """)),
                html.Pre(id="click-data", style=styles["pre"]),
            ], className="three columns"),

            html.Div([
                dcc.Markdown(dedent("""
                    **Selection Data**

                    Choose the lasso or rectangle tool in the graph's menu
                    bar and then select points in the graph.

                    Note that if `layout.clickmode = 'event+select'`, selection data also
                    accumulates (or un-accumulates) selected data if you hold down the shift
                    button while clicking.
                """)),
                html.Pre(id="selected-data", style=styles["pre"]),
            ], className="three columns"),

            html.Div([
                dcc.Markdown(dedent("""
                    **Zoom and Relayout Data**

                    Click and drag on the graph to zoom or click on the zoom
                    buttons in the graph's menu bar.
                    Clicking on legend items will also fire
                    this event.
                """)),
                html.Pre(id="relayout-data", style=styles["pre"]),
            ], className="three columns")
        ]),
    ], style={
        "borderBottom": "thin lightgrey solid",
        "backgroundColor": "rgb(250, 250, 250)",
        "padding": "10px 5px",
        "width": "49%",
        "display": "inline-block"
    }),

    html.Div([
        dcc.RadioItems(
            id="crypto_radio",
            options=[{"label": i, "value": i} for i in Env.bases],
            labelStyle={"display": "inline-block"}
        ),
        dcc.Checklist(
            id="indicator_select",
            options=[{"label": i, "value": i} for i in indicator_opts],
            value=[],
            labelStyle={"display": "inline-block"}
        ),
        dcc.Graph(id="graph4h"),
        # dcc.Graph(id="volume-signals-graph"),
        html.Div(id="graph4h_end", style={"display": "none"}),
        dash_table.DataTable(
            id="kpi_table", editable=False,  # hidden_columns=["id"],
            style_header={
                "backgroundColor": "rgb(230, 230, 230)",
                "fontWeight": "bold"
            }
        )
        # html.Div(id="kpi_table_wrapper")
    ], style={"display": "inline-block", "float": "right", "width": "49%"}),

])


def df_check(df, timerange):
    if timerange is not None:
        logger.debug(f"timerange: {timerange} len(df): {len(df)}")
    else:
        logger.debug(f"timerange: all len(df): {len(df)}")
    logger.debug(str(df.head(1)))
    logger.debug(str(df.tail(1)))


def get_trigger_update_data(update_data_click):
    ctx = dash.callback_context
    if not ctx.triggered:
        trigger = "No Id"
    else:
        # logger.debug(f"triggered: {ctx.triggered[0]}")
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    # logger.debug(f"trigger: {trigger}")

    if (update_data_click is not None) and (trigger == "update_data"):
        # for base in ohlcv_df_dict.keys():
        #     del ohlcv_df_dict[base]
        global ohlcv_df_dict
        ohlcv_df_dict = None

        ohlcv = ccd.Ohlcv()
        bases = Env.bases
        # data_objs = uch.all_data_objs(ohlcv)
        uch.update_to_now(bases, ohlcv, [ohlcv])
        ohlcv_df_dict = {base: ohlcv.load_data(base) for base in Env.bases}
    return trigger


@app.callback(
    dash.dependencies.Output("focus", "children"),
    [dash.dependencies.Input("graph_all", "clickData"),
     dash.dependencies.Input("graph6month", "clickData"),
     dash.dependencies.Input("graph10day", "clickData"),
     dash.dependencies.Input("graph1day", "clickData"),
     dash.dependencies.Input("graph4h", "clickData"),
     dash.dependencies.Input("update_data", "n_clicks_timestamp")],
    [dash.dependencies.State("focus", "children")])
def set_focus_time(click_all, click_6month, click_10day, click_1day, click_4h, update_data_click, focus_json):
    graph_position = None
    if focus_json is not None:
        graph_position = json.loads(focus_json)
    if graph_position is None:
        graph_position = dict()
        for g in ["graph_all", "graph6month", "graph10day", "graph1day", "graph4h"]:
            graph_position[g] = dict()
            graph_position[g]["focus"] = None
            graph_position[g]["end"] = None

    trigger = get_trigger_update_data(update_data_click)
    graph_list = []

    if trigger == "graph_all":
        click_data = click_all["points"][0]["x"] if click_all is not None else None
        graph_position["graph_all"]["focus"] = click_data
        graph_list = ["graph6month", "graph10day", "graph1day", "graph4h"]
    elif trigger == "graph6month":
        click_data = click_6month["points"][0]["x"] if click_6month is not None else None
        graph_position["graph6month"]["focus"] = click_data
        graph_list = ["graph10day", "graph1day", "graph4h"]
    elif trigger == "graph10day":
        click_data = click_10day["points"][0]["x"] if click_10day is not None else None
        graph_position["graph10day"]["focus"] = click_data
        graph_list = ["graph1day", "graph4h"]
    elif trigger == "graph1day":
        click_data = click_1day["points"][0]["x"] if click_1day is not None else None
        graph_position["graph1day"]["focus"] = click_data
        graph_list = ["graph4h"]
    elif trigger == "graph4h":
        click_data = click_4h["points"][0]["x"] if click_4h is not None else None
        graph_position["graph4h"]["focus"] = click_data
    else:
        click_data = None
        graph_list = ["graph_all", "graph6month", "graph10day", "graph1day", "graph4h"]
    for g in graph_list:
        graph_position[g]["focus"] = None
        graph_position[g]["end"] = click_data
    # logger.debug(f"set_focus_time: triggered: {trigger} at {click_data} graph_positions \n{json.dumps(graph_position, indent=4)}")  # noqa

    return json.dumps(graph_position)


def get_end_focus(focus_json: str, base_radio: str, graph: str):
    """ returns (focus, end, focus_is_end), i.e.

        - the focus timestamp if there was a click,
        - the end timestamp of base_radio or now if there is no base_radio,
        - a boolean dict with graph names as key that is True if the focus is equal to end and
          that is False if a focus mark shall be shown in the graph.
    """
    end = None
    if focus_json is not None:
        graph_position = json.loads(focus_json)
        if (graph_position is not None) and (graph in graph_position):
            focus = graph_position[graph]["focus"]
            if focus is not None:
                focus = pd.Timestamp(focus, tz=Env.tz)
            end = graph_position[graph]["end"]
            if end is not None:
                end = pd.Timestamp(end, tz=Env.tz)
        else:
            focus = None
    else:
        focus = None
    if (end is None) or (end is pd.NaT):
        if base_radio is not None:
            end = ohlcv_df_dict[base_radio].index[-1]
    if (focus is not None) and (focus is pd.NaT):
        focus = None
    return (focus, end)


@app.callback(
    dash.dependencies.Output("graph_all", "figure"),
    [dash.dependencies.Input("focus", "children"),
     dash.dependencies.Input("crypto_select", "value"),
     dash.dependencies.Input("crypto_radio", "value")])
def update_graph(focus_json, bases, base_radio):
    graph_bases = []
    (focus, _) = get_end_focus(focus_json, None, "graph_all")
    if (focus is not None):
        x1 = focus - pd.Timedelta(365/2, "D")
        x2 = focus
        graph_bases.append(
            dict(x=[x1, x2, x2, x1, x1], y=[0, 0, 1, 1, 0],
                 mode="none", yaxis="y", fill="tonexty", color="#bee4e7"))
    if bases is None:
        bases = []
    if base_radio not in bases:
        bases.append(base_radio)
    for base in bases:
        # df_check(ohlcv_df_dict[base], None)
        bmd = ohlcv_df_dict[base].resample("D").agg({"close": "first"})  # bmd == base minute data
        # bmd = ohlcv_df_dict[base].resample("D").agg({"open": "first", "close": "last", "high": "max",
        #                                     "low": "min", "volume": "sum"})
        bmd = bmd/bmd.max()  # normalize
        graph_bases.append(dict(
            x=bmd.index,
            y=bmd["close"],
            mode="lines",  # "lines+markers"
            name=base
        ))
    return {
        "data": graph_bases,
        "layout": {
            "height": 350,
            "margin": {"l": 20, "b": 30, "r": 10, "t": 10},
            "annotations": [{
                "x": 0, "y": 0.85, "xanchor": "left", "yanchor": "bottom",
                "xref": "paper", "yref": "paper", "showarrow": False,
                "align": "left", "bgcolor": "rgba(255, 255, 255, 0.5)",
                "text": "normalized crypto prices"
            }],
            "yaxis": {"type": "linear"},
            "xaxis": {"showgrid": False},
            "showlegend": False
        }
    }


def regression_gradient_distance(start, end, bmd):
    """ returns the regression line gradient and
        the y distance of the last y data point from the regression line
    """
    reduced_bmd = bmd.loc[(bmd.index >= start) & (bmd.index <= end)]
    if reduced_bmd.empty:
        logger.warning(f"empty df for bmd len: {len(bmd)} bmd end: {bmd.index[-1]} start: {start} end: {end}")
        gradient = distance = 0
    else:
        gradient, Y_pred = ind.time_linear_regression(reduced_bmd["close"])
        distance = reduced_bmd["close"].values.reshape(-1, 1)[-1] - Y_pred[-1]
    return (gradient, distance)


def volume_relation_check(short_range, long_range, end, ohlcv_df):
    short_start = end - short_range
    long_start = end - long_range
    sdf = ohlcv_df.loc[(ohlcv_df.index >= short_start) & (ohlcv_df.index <= end)]
    ldf = ohlcv_df.loc[(ohlcv_df.index >= long_start) & (ohlcv_df.index <= end)]
    vol_rel = sdf["volume"].mean() / ldf["volume"].mean()
    # logger.debug(f"vol_rel: {vol_rel} = sdf: {sdf.volume.mean()}/ ldf: {ldf.volume.mean()}")
    return vol_rel


def minute_liquidity(timerange, end, ohlcv_df):
    start = end - timerange
    sdf = ohlcv_df.loc[(ohlcv_df.index >= start) & (ohlcv_df.index <= end)]
    sdf.loc[:, "liq"] = sdf["volume"] * sdf["close"]
    liquidity = sdf["liq"].median()
    return liquidity


@app.callback(
    [dash.dependencies.Output("kpi_table", "data"),
     dash.dependencies.Output("kpi_table", "columns"),
     dash.dependencies.Output("kpi_table", "style_data_conditional")],
    [dash.dependencies.Input("focus", "children"),
     dash.dependencies.Input("crypto_select", "value"),
     dash.dependencies.Input("crypto_radio", "value")])
def update_table(focus_json, bases, base_radio):
    (focus, end) = get_end_focus(focus_json, None, "graph4h")
    if focus is not None:
        end = focus
    df = pd.DataFrame(index=Env.bases)
    df.index.rename("base", inplace=True)
    regr_config = view_config["kpi_table"]["regression"]
    vol_config = view_config["kpi_table"]["volume"]
    liq_config = view_config["kpi_table"]["liquidity"]
    max_timerange = max([regr_config[reg]["timerange"] for reg in regr_config])
    for base in Env.bases:
        if end is None:  # is the case for initialization
            rend = ohlcv_df_dict[base].index[-1]
        else:
            rend = end  # different and per base possible -> don't change end but introduce rend
        start = rend - max_timerange
        bmd = normalize_close(ohlcv_df_dict[base], start, rend, "T")
        if (bmd is not None) and (not bmd.empty):
            for regression in regr_config:
                start = rend - regr_config[regression]["timerange"]
                gradient, distance = regression_gradient_distance(start, rend, bmd)
                df.loc[base, regression+"grad"] = gradient
                df.loc[base, regression+"dist"] = distance
            for vol in vol_config:
                df.loc[base, vol+"volrel"] = volume_relation_check(
                    vol_config[vol]["shortrange"], vol_config[vol]["longrange"], rend, ohlcv_df_dict[base])
            for liq in liq_config:
                df.loc[base, liq+"_liq"] = minute_liquidity(
                    liq_config[liq]["timerange"], rend, ohlcv_df_dict[base])
    gl = [regression+"grad" for regression in regr_config]
    dl = [regression+"dist" for regression in regr_config]
    vl = [vol+"volrel" for vol in vol_config]
    ll = [liq+"_liq" for liq in liq_config]
    cmp = dict()
    cmp["buy_grad"] = {
        regression+"grad": regr_config[regression]["buy"]
        for regression in regr_config}
    cmp["sell_grad"] = {
        regression+"grad": regr_config[regression]["sell"]
        for regression in regr_config}
    cmp["buy_dist"] = {regression+"dist": -0.5 for regression in regr_config}
    cmp["sell_dist"] = {regression+"dist": 0.5 for regression in regr_config}
    cmp["best_grad"] = {
        col: max([cmp["buy_grad"][col]*2, value])
        for (col, value) in df.quantile(0.9).iteritems() if "grad" in col}
    cmp["best_dist"] = {
        col: min([cmp["buy_dist"][col]*2, value])
        for (col, value) in df.quantile(0.9).iteritems() if "dist" in col}
    cmp["worst_grad"] = {
        col: min([cmp["sell_grad"][col]*2, value])
        for (col, value) in df.quantile(0.1).iteritems() if "grad" in col}
    cmp["worst_dist"] = {
        col: max([cmp["sell_dist"][col]*2, value])
        for (col, value) in df.quantile(0.1).iteritems() if "dist" in col}
    # logger.debug(json.dumps(cmp, indent=4))
    df = df[gl+dl+vl+ll]
    df = df.reset_index()
    df["id"] = df["base"]
    df = df.sort_values(ll[0], ascending=False)
    buy_grad = [
        {
            "if": {
                "filter_query": "{{{}}} >= {}".format(col, cmp["buy_grad"][col]),
                "column_id": col
            },
            "backgroundColor": "#86cb66",
            "color": "black"
        } for col in cmp["buy_grad"]
    ]
    buy_dist = [
        {
            "if": {
                "filter_query": "{{{}}} <= {}".format(col, cmp["buy_dist"][col]),
                "column_id": col
            },
            "backgroundColor": "#86cb66",
            "color": "black"
        } for col in cmp["buy_dist"]
    ]
    sell_grad = [
        {
            "if": {
                "filter_query": "{{{}}} <= {}".format(col, cmp["sell_grad"][col]),
                "column_id": col
            },
            "backgroundColor": "#f24632",
            "color": "black"
        } for col in cmp["sell_grad"]
    ]
    sell_dist = [
        {
            "if": {
                "filter_query": "{{{}}} >= {}".format(col, cmp["sell_dist"][col]),
                "column_id": col
            },
            "backgroundColor": "#f24632",
            "color": "black"
        } for col in cmp["sell_dist"]
    ]
    best_grad = [
        {
            "if": {
                "filter_query": "{{{}}} >= {}".format(col, cmp["best_grad"][col]),
                "column_id": col
            },
            "backgroundColor": "#006837",
            "color": "white"
        } for col in cmp["best_grad"]
    ]
    best_dist = [
        {
            "if": {
                "filter_query": "{{{}}} <= {}".format(col, cmp["best_dist"][col]),
                "column_id": col
            },
            "backgroundColor": "#006837",
            "color": "white"
        } for col in cmp["best_dist"]
    ]
    worst_grad = [
        {
            "if": {
                "filter_query": "{{{}}} <= {}".format(col, cmp["worst_grad"][col]),
                "column_id": col
            },
            "backgroundColor": "#a50026",
            "color": "white"
        } for col in cmp["worst_grad"]
    ]
    worst_dist = [
        {
            "if": {
                "filter_query": "{{{}}} >= {}".format(col, cmp["worst_dist"][col]),
                "column_id": col
            },
            "backgroundColor": "#a50026",
            "color": "white"
        } for col in cmp["worst_dist"]
    ]
    return [
        df.to_dict("records"),
        [{
            "id": id, "name": id, "type": "numeric",
            "format": Format(precision=5, scheme=Scheme.fixed)} for id in df.columns],
        [
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "rgb(248, 248, 248)"
            },
            {
                "if": {
                    "filter_query": "{{{}}} = {}".format("base", base_radio),
                },
                "backgroundColor": "#fafa6e",
                "color": "inherit"
            },
            *buy_grad,
            *sell_grad,
            *buy_dist,
            *sell_dist,
            *best_grad,
            *worst_grad,
            *best_dist,
            *worst_dist,
        ]
    ]


@app.callback(
    Output("crypto_select", "value"),
    [Input("all_button", "n_clicks_timestamp"), Input("none_button", "n_clicks_timestamp")])
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
    Output("crypto_radio", "value"),
    [Input("crypto_select", "value"),
     Input("kpi_table", "active_cell")],
    [State("crypto_radio", "value")])
def radio_range(selected_cryptos, active_cell, base_radio):
    if active_cell is not None:
        return active_cell["row_id"]
    if (selected_cryptos is not None) and (len(selected_cryptos) > 0):
        if base_radio in selected_cryptos:
            return base_radio
        else:
            return selected_cryptos[0]
    else:
        return Env.bases[0]


def one_result(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


@app.callback(
    Output("hover-data", "children"),
    [
     Input("graph4h", "hoverData"),
     Input("graph1day", "hoverData"),
     Input("graph10day", "hoverData"),
     Input("graph6month", "hoverData"),
     Input("graph_all", "hoverData")
    ])
def display_hover_data(*args):
    interaction_data = one_result(args)
    return json.dumps(interaction_data, indent=2)


@app.callback(
    Output("click-data", "children"),
    [
     Input("graph4h", "clickData"),
     Input("graph1day", "clickData"),
     Input("graph10day", "clickData"),
     Input("graph6month", "clickData"),
     Input("graph_all", "clickData")
    ])
def display_click_data(*args):
    interaction_data = one_result(args)
    return json.dumps(interaction_data, indent=2)


@app.callback(
    Output("selected-data", "children"),
    [
     Input("graph4h", "selectedData"),
     Input("graph1day", "selectedData"),
     Input("graph10day", "selectedData"),
     Input("graph6month", "selectedData"),
     Input("graph_all", "selectedData")
    ])
def display_selected_data(*args):
    interaction_data = one_result(args)
    return json.dumps(interaction_data, indent=2)


@app.callback(
    Output("relayout-data", "children"),
    [
     Input("graph4h", "relayoutData"),
     Input("graph1day", "relayoutData"),
     Input("graph10day", "relayoutData"),
     Input("graph6month", "relayoutData"),
     Input("graph_all", "relayoutData")
    ])
def display_relayout_data(*args):
    interaction_data = one_result(args)
    return json.dumps(interaction_data, indent=2)


def normalize_close(bmd, start, end, aggregation):
    bmd = bmd.loc[(bmd.index >= start) & (bmd.index <= end)]
    if aggregation != "T":
        bmd = bmd.resample(aggregation).agg({"close": "first"})

    if (bmd is not None) and (len(bmd) > 0):
        normfactor = bmd.iloc[-1].close
        bmd = bmd.apply(lambda x: (x / normfactor - 1) * 100)  # normalize to % change
    return bmd


def normalize_ohlc(bmd, start, end, aggregation):
    bmd = bmd.loc[(bmd.index >= start) & (bmd.index <= end)]
    if aggregation != "T":
        bmd = bmd.resample(aggregation).agg({"open": "first", "close": "last", "high": "max",
                                             "low": "min"})  # "volume": "sum"
    if (bmd is not None) and (len(bmd) > 0):
        normfactor = bmd.iloc[-1].close
        bmd = bmd.apply(lambda x: (x / normfactor - 1) * 100)  # normalize to % change
    return bmd


def volume_graph(start, end, bmd):
    bmd = bmd.loc[(bmd.index >= start) & (bmd.index <= end)]
    INCREASING_COLOR = "#17BECF"
    DECREASING_COLOR = "#7F7F7F"
    colors = []
    for i in range(len(bmd.close)):
        if i != 0:
            if bmd.close[i] > bmd.close[i-1]:
                colors.append(INCREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        else:
            colors.append(DECREASING_COLOR)
    return dict(x=bmd.index, y=bmd.volume, marker=dict(color=colors), type="bar", yaxis="y2", name="Volume")


def straigt_line(start, end, end_y, gain_per_hour, legendname):
    """ Input: an 'end' timestamp with the corresponding y value 'end_y' and the gain per hour.

        Returns the plotly straight line graph between 'start' and 'end'.
    """
    start_y = end_y - float((end - start) / pd.Timedelta(60, "T")) * gain_per_hour
    return dict(x=[start, end], y=[start_y, end_y], mode="lines", name=legendname, yaxis="y")


def chance_risk_marker(start, end, end_y, chance, risk, legendname):
    """ Input: an 'end' timestamp with the corresponding y value 'end_y' and the chance and risk.

        Returns the plotly straight marker graph at 'end'.
    """
    return dict(x=[end, end], y=[end_y+chance, end_y-risk], mode="marker", name=legendname, yaxis="y")


def volume_ratio_marker(time, vol_ratio, legendname):
    """ Input: a 'time' timestamp with the corresponding y value 'vol_rel'.

        Returns the plotly straight marker graph.
    """
    return dict(x=[time], y=[vol_ratio], mode="scatter", name=legendname, yaxis="y")


def show_condensed_features(base, start, time):
    """ Input 'time' timestamp of period and base minute data 'bmd' for which features are required.
        'bmd' shall have history data of cof.MHE minutes before 'time' for feature calculation.

        Returns a plotly graph that visualizes the features at timestamp 'time'. The graph starts at 'start'.
    """
    logger.warning("timely disabled")
    return None

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
        # time_regr_y = float(bf_df["price"+ext])
        gain = float(bf_df["gain"+ext])
        line_end = time - pd.Timedelta(offset, "T")
        line_start = line_end - pd.Timedelta(minutes-1, "T")
        line_start = max(line_start, start)
        time_cur_y = float(red_bmdc.loc[line_end, "close"])
        legendname = "{}: d/h = {:4.3f}".format(ext, gain)
        # logger.debug(f"straigt_line({line_start}, {line_end}, {time_cur_y}, {gain}, {legendname})")
        yield straigt_line(line_start, line_end, time_cur_y, gain, legendname)
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
    # ! ERROR ind.time_linear_regression expect aggregation = "m", which is not the case for graph6month
    delta, Y_pred = ind.time_linear_regression(reduced_bmd["close"])
    aggmin = (end - start) / pd.Timedelta(1, aggregation)
    legendname = "{:4.0f} {} = delta/h: {:4.3f}".format(aggmin, aggregation, delta)
    return dict(x=reduced_bmd.index[[0, -1]], y=Y_pred, mode="lines", name=legendname, yaxis="y")


def timeline_graph(graph: str, focus, end, bases, base_radio, indicators):
    """ Displays a line chart of close prices
        and a one dimensional close prices regression line both on the same Datetimeindex.
    """
    aggregation = view_config[graph]["aggregation"]
    graph_bases = []
    start = end - view_config[graph]["timerange"]

    if focus is not None:
        x1 = focus - view_config[graph]["focusrange"]
        x2 = focus
        graph_bases.append(
            dict(x=[x1, x2, x2, x1, x1], y=[0, 0, 1, 1, 0],
                 mode="none", yaxis="y2", fill="tonexty", color="#bee4e7"))

    if bases is None:
        bases = []
    for base in Env.bases:
        if (base in bases) or (base == base_radio):
            if (start is None) or (end is None) or (aggregation is None):
                logger.warning(f"start: {start}, end: {end}, aggregation: {aggregation}")
            if end > ohlcv_df_dict[base].index[0]:
                bmd = normalize_close(ohlcv_df_dict[base], start, end, aggregation)
                regr = regression_graph(start, end, bmd, aggregation)
                if indicators is None:
                    indicators = []
                else:
                    if "regression 1D" in indicators:
                        if base == base_radio:
                            graph_bases.append(regr)

                legendname = base + " " + regr["name"]
                graph_bases.append(dict(x=bmd.index, y=bmd["close"],  mode="lines", name=legendname))
            else:
                logger.debug(f"{base} start {ohlcv_df_dict[base].index[0]} is later than selected end {end}")

    timeinfo = aggregation + ": " + start.strftime(Env.dt_format) + " - " + end.strftime(Env.dt_format)
    return {
        "data": graph_bases,
        "layout": {
            "height": 350,
            "margin": {"l": 20, "b": 30, "r": 10, "t": 10},
            "annotations": [{
                "x": 0, "y": 0.85, "xanchor": "left", "yanchor": "bottom",
                "xref": "paper", "yref": "paper", "showarrow": False,
                "align": "left", "bgcolor": "rgba(255, 255, 255, 0.5)",
                "text": "normalized crypto prices"
            }],
            "yaxis": {"type": "linear"},
            "xaxis": {"showgrid": False, "title": timeinfo},
            "yaxis2": {"side": "right", "overlaying": "y", "anchor": "x", "visible": False},
            "showlegend": True
        }
    }


def dash_trigger():
    ctx = dash.callback_context
    if not ctx.triggered:
        trigger = "No Id"
    else:
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    return trigger


@app.callback(
    dash.dependencies.Output("graph6month", "figure"),
    [dash.dependencies.Input("focus", "children"),
     dash.dependencies.Input("crypto_select", "value"),
     dash.dependencies.Input("crypto_radio", "value"),
     dash.dependencies.Input("indicator_select", "value")])
def update_graph6month(focus_json, bases, base_radio, indicators):
    (focus, end) = get_end_focus(focus_json, base_radio, "graph6month")
    return timeline_graph("graph6month", focus, end, bases, base_radio, indicators)


@app.callback(
    dash.dependencies.Output("graph10day", "figure"),
    [dash.dependencies.Input("focus", "children"),
     dash.dependencies.Input("crypto_select", "value"),
     dash.dependencies.Input("crypto_radio", "value"),
     dash.dependencies.Input("indicator_select", "value")])
def update_graph10day(focus_json, bases, base_radio, indicators):
    (focus, end) = get_end_focus(focus_json, base_radio, "graph10day")
    return timeline_graph("graph10day", focus, end, bases, base_radio, indicators)


@app.callback(
    dash.dependencies.Output("graph1day", "figure"),
    [dash.dependencies.Input("focus", "children"),
     dash.dependencies.Input("crypto_select", "value"),
     dash.dependencies.Input("crypto_radio", "value"),
     dash.dependencies.Input("indicator_select", "value")])
def update_graph1day(focus_json, bases, base_radio, indicators):
    (focus, end) = get_end_focus(focus_json, base_radio, "graph1day")
    return timeline_graph("graph1day", focus, end, bases, base_radio, indicators)


def target_list(base, start, end, bmd):
    logger.warning("timely disabled")
    return None

    # labels1 = [i for i in range(241)]
    # logger.debug(f"{len(labels1)}, {labels1}")
    # return labels1, None
    colormap = {ct.TARGETS[ct.HOLD]: 0, ct.TARGETS[ct.BUY]: 1, ct.TARGETS[ct.SELL]: -1}

    target_dict = dict()
    signals_df = classifier_set.predict_signals(base, start, end)
    target_colors = [colormap[t] for t in signals_df["signal"]]
    labels = [ct.TARGET_NAMES[t] for t in signals_df["signal"]]
    target_dict["signals"] = {"targets": target_colors, "labels": labels}

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
    target_dict["target"] = {"targets": target_colors, "labels": labels}

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
        yaxis="y3", name="labels",
        text=[target_dict[t]["labels"] for t in target_dict],
        colorscale="RdYlGn",
        reversescale=False,
        hoverinfo="x+y+z+text+name",
        showscale=False,
        # autocolorscale=False,
        # colorscale=[[cf.TARGETS[cf.HOLD]/2, "rgb(255, 234, 0)"],
        #             [cf.TARGETS[cf.SELL]/2, "rgb(255, 0, 0)"],
        #             [cf.TARGETS[cf.BUY]/2, "rgb(116, 196, 118)"]],
        colorbar=dict(tick0=0, dtick=1))


def ymin_ymax(start, end):
    max = min = 0
    for base in Env.bases:
        bmd = normalize_close(ohlcv_df_dict[base], start, end, "T")
        bmd = bmd.drop(columns=["volume"])
        bmax = bmd.max().max()
        if bmax > max:
            max = bmax
        bmin = bmd.min().min()
        if bmin < min:
            min = bmin
    return min, max


@app.callback(
    dash.dependencies.Output("graph4h", "figure"),
    [dash.dependencies.Input("focus", "children"),
     dash.dependencies.Input("crypto_select", "value"),
     dash.dependencies.Input("crypto_radio", "value"),
     dash.dependencies.Input("indicator_select", "value")])
def update_detail_graph_by_click(focus_json, bases, base, indicators):
    """ Displays candlestick charts of the selected time and max time aggs back
        together with selected indicators
    """
    (focus, end) = get_end_focus(focus_json, base, "graph4h")

    graph_bases = []

    aggregation = view_config["graph4h"]["aggregation"]
    if indicators is None:
        indicators = []
    start = end - view_config["graph4h"]["timerange"]
    ymin, ymax = ymin_ymax(start, end)

    bmd = ohlcv_df_dict[base]
    if end > bmd.index[0]:
        nbmd = normalize_ohlc(bmd, start, end, aggregation)
        if "equal scale" in indicators:
            ymin, ymax = ymin_ymax(start, end)
        else:
            scale_nbmd = nbmd.drop(columns=["volume"])
            ymax = scale_nbmd.max().max()
            ymin = scale_nbmd.min().min()

        if "targets" in indicators:
            target_dict = target_list(base, start, end, bmd)
            graph_bases.append(
                go.Candlestick(x=nbmd.index, open=nbmd.open, high=nbmd.high, low=nbmd.low,
                               close=nbmd.close, yaxis="y",
                               hoverinfo="x+y+z+text+name", text=target_dict["target"]["labels"]))
            graph_bases.append(target_heatmap(base, start, end, nbmd, target_dict))
        else:
            graph_bases.append(
                go.Candlestick(x=nbmd.index, open=nbmd.open, high=nbmd.high, low=nbmd.low,
                               close=nbmd.close, yaxis="y",
                               hoverinfo="x+y+z+text+name"))

        if "regression 1D" in indicators:
            for rline in view_config["graph4h"]["regression"]:
                rstart = end - view_config["graph4h"]["regression"][rline]["timerange"]
                graph_bases.append(
                    regression_graph(rstart, end, nbmd, aggregation))

        if focus is not None:
            graph_bases.append(dict(x=[focus, focus], y=[0, 1], mode="lines", yaxis="y4"))

        graph_bases.append(volume_graph(start, end, bmd))

        if ("features" in indicators) and (focus is not None):
            for graph in show_condensed_features(base, start, focus):
                graph_bases.append(graph)
    else:
        logger.debug(f"{base} start {bmd.index[0]} is later than selected end {end}")
    timeinfo = aggregation + ": " + start.strftime(Env.dt_format) + " - " + end.strftime(Env.dt_format)
    return {
        "data": graph_bases,
        "layout": {
            "height": 500,
            "margin": {"l": 20, "b": 30, "r": 10, "t": 10},
            "annotations": [{
                "x": 0, "y": 0.85, "xanchor": "left", "yanchor": "bottom",
                "xref": "paper", "yref": "paper", "showarrow": False,
                "align": "left", "bgcolor": "rgba(255, 255, 255, 0.5)",
                "text": "normalized crypto prices"
            }],
            "yaxis2": {"domain": [0., 0.2], "side": "left"},
            "yaxis4": {"domain": [0., 0.2], "side": "right", "overlaying": "y"},
            "yaxis3": {"domain": [0.2, 0.23], "showticklabels": False},
            "yaxis": {"type": "linear", "domain": [0.3, 1], "range": [ymin, ymax]},
            "xaxis": {"showgrid": False, "title": timeinfo, "rangeslider": {"visible": False}}
        }
    }


# wish list:
# show targets in history views
# show performance transaction start/stop in history view
# visualize chance and risk features in 4h graph
# detecting volume emerging new cryptos
# KPI table of all focus cryptos (above liquidity criteria)
if __name__ == "__main__":
    # load_crypto_data()
    # env.test_mode()
    tee = env.Tee(log_prefix="CryptoDashboard")
    ohlcv = ccd.Ohlcv()
    features = cof.F3cond14(ohlcv)
    targets = ct.Target10up5low30min(ohlcv)
    ohlcv_df_dict = {base: ohlcv.load_data(base) for base in Env.bases}
    # ! temporary disabled classifier_set = cp.ClassifierSet()
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run_server(debug=True, dev_tools_silence_routes_logging=False)
