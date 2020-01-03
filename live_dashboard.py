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
        dcc.Graph(id="half-day-time-graph"),
        dcc.Graph(id="ten-day-time-graph"),
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
        dcc.Graph(id='y-time-series'),
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
    dash.dependencies.Output('ten-day-time-graph', 'figure'),
    [dash.dependencies.Input('full-time-graph', 'clickData'),
     dash.dependencies.Input('crossfilter-crypto-select', 'value'),
     dash.dependencies.Input('crossfilter-indicator-select', 'value')])
def update_ten_day_by_click(click_data, bases, indicators):
    """ Displays candlestick charts of the selected time and max time aggs back
        together with selected indicators
    """
    end = pd.Timestamp.now(tz='UTC')
    if bases is not None and len(bases) > 0:
        if click_data is None:
            end = cdf[bases[0]].index[len(cdf[bases[0]])-1]
        else:
            end = pd.Timestamp(click_data['points'][0]['x'], tz='UTC')
    start = end - pd.Timedelta(Env.minimum_minute_df_len, "m")
    if bases is None:
        bases = []
    if indicators is None:
        indicators = []
    # print("zoom-in", bases, indicators, start, end)
    return update_ten_day_graph(bases, indicators, start, end)


def update_ten_day_graph(bases, indicators, start, end):
    graph_bases = []
    aggregation = "5T"
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


@app.callback(
    dash.dependencies.Output('half-day-time-graph', 'figure'),
    [dash.dependencies.Input('ten-day-time-graph', 'clickData'),
     dash.dependencies.Input('crossfilter-crypto-select', 'value'),
     dash.dependencies.Input('crossfilter-indicator-select', 'value')])
def update_half_day_by_click(click_data, bases, indicators):
    """ Displays candlestick charts of the selected time and max time aggs back
        together with selected indicators
    """
    end = pd.Timestamp.now(tz='UTC')
    if bases is not None and len(bases) > 0:
        if click_data is None:
            end = cdf[bases[0]].index[len(cdf[bases[0]])-1]
        else:
            end = pd.Timestamp(click_data['points'][0]['x'], tz='UTC')
    start = end - pd.Timedelta(12*60, "m")
    if bases is None:
        bases = []
    if indicators is None:
        indicators = []
    # print("zoom-in", bases, indicators, start, end)
    return update_half_day_graph(bases, indicators, start, end)


def update_half_day_graph(bases, indicators, start, end):
    graph_bases = []
    aggregation = "T"
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


@app.callback(
    dash.dependencies.Output('zoom-in-graph', 'figure'),
    [dash.dependencies.Input('full-time-graph', 'clickData'),
     dash.dependencies.Input('crossfilter-crypto-radio', 'value'),
     dash.dependencies.Input('crossfilter-indicator-select', 'value')])
def update_detail_graph_by_click(click_data, base, indicators):
    """ Displays candlestick charts of the selected time and max time aggs back
        together with selected indicators
    """
    end = pd.Timestamp.now(tz='UTC')
    if base is not None:
        print(base)
        if click_data is None:
            end = cdf[base].index[len(cdf[base])-1]
        else:
            end = pd.Timestamp(click_data['points'][0]['x'], tz='UTC')
    start = end - pd.Timedelta(4*60, "m")
    if indicators is None:
        indicators = []
    # print("zoom-in", bases, indicators, start, end)
    return update_detail_graph(base, indicators, start, end)


def update_detail_graph(base, indicators, start, end):
    graph_bases = []
    aggregation = "T"
    timeinfo = aggregation + ": " + start.strftime(Env.dt_format) + " - " + end.strftime(Env.dt_format)
    dcdf = cdf[base]
    dcdf = dcdf.loc[(dcdf.index >= start) & (dcdf.index <= end)]
    dcdf = dcdf.resample(aggregation).agg({"open": "first", "close": "last", "high": "max",
                                           "low": "min", })  # "volume": "sum"
    if (dcdf is not None) and (len(dcdf) > 0):
        normfactor = dcdf.iloc[0].open
        dcdf = dcdf.apply(lambda x: (x / normfactor - 1) * 100)  # normalize to % change
    graph_bases.append(
        go.Candlestick(x=dcdf.index,
                       open=dcdf.open,
                       high=dcdf.high,
                       low=dcdf.low,
                       close=dcdf.close))
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


def load_crypto_data():
    # cdf = {base: ccd.load_asset_dataframe(base) for base in Env.usage.bases}
    [print(base, len(cdf[base])) for base in cdf]
    print("xrp minute  aggregated", cdf["xrp"].head())
    dcdf = cdf["xrp"].resample("D").agg({"open": "first", "close": "last", "high": "max",
                                         "low": "min", "volume": "sum"})
    print("xrp day aggregated", dcdf.head())


if __name__ == '__main__':
    # load_crypto_data()
    cdf = {base: ccd.load_asset_dataframe(base, path=Env.data_path) for base in Env.usage.bases}
    app.run_server(debug=True)
