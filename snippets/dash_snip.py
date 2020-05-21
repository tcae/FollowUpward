import dash
import dash_html_components as html
import dash_colorscales
import json

app = dash.Dash('')

app.scripts.config.serve_locally = True

app.layout = html.Div([
    dash_colorscales.DashColorscales(
        id='colorscale-picker',
        # nSwatches=7,
        fixSwatches=False
    ),
    html.P(id='output', children='')
])


@app.callback(
        dash.dependencies.Output('output', 'children'),
        [dash.dependencies.Input('colorscale-picker', 'colorscale')])
def display_output(colorscale):
    return json.dumps(colorscale)


if __name__ == '__main__':
    app.run_server(debug=True)
