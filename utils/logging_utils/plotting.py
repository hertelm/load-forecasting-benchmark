import typing as t
from collections.abc import Iterable

import wandb

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
# import colorcet

import numpy as np

custom_template = go.layout.Template(
    layout = {
        'template': 'plotly_white',
        'width':1240,
        'height':750,
        # 'colorscale': {
        #     'diverging': colorcet.CET_D1A,
        # },
        'colorway': px.colors.qualitative.T10,
        'font': {
            'size': 20,
            'color': 'black'
            },
        'xaxis': {
            'linewidth': 1,
            'linecolor': 'black',
            'title': ' ',
            'mirror': True,
            'ticks': 'outside',
            'showline': True,
            'gridwidth': 0,
            },
        'yaxis': {
            'linewidth': 1,
            'linecolor': 'black',
            'title': {
                'text': ' ',
                'standoff': 40,
            },
            'mirror': True,
            'ticks': 'outside',
            'showline': True,
            'gridwidth': 0,

        },
        'legend': {
            'font': {
                'size': 20,
            },
            'orientation': 'h',
            'title': ''
        },
        'margin': {
            't': 50,
            'r': 50,
            'l': 150,
            'b': 100,
        },
        'title': {
            'xanchor': 'center',
            'x': 0.5,
        },
    }
)

pio.templates.default = custom_template

def sample_plot(
    xs: t.Union[t.Iterable, t.Iterable[t.Iterable]],
    ys: t.Iterable[t.Iterable],
    keys: t.Optional[t.Iterable] = None,
    title: t.Optional[str] = None,
    xname: t.Optional[str] = None,
):
    """Construct a line series plot.

    Arguments:
        xs (array of arrays, or array): Array of arrays of x values
        ys (array of arrays): Array of y values
        keys (array): Array of labels for the line plots
        title (string): Plot title.
        xname: Title of x-axis

    Returns:
        A plot object, to be passed to wandb.log()

    Example:
        When logging a singular array for xs, all ys are plotted against that xs
        <!--yeadoc-test:plot-line-series-single-->
        ```python
        import wandb

        run = wandb.init()
        xs = [i for i in range(10)]
        ys = [[i for i in range(10)], [i**2 for i in range(10)]]
        run.log(
            {"line-series-plot1": wandb.plot.line_series(xs, ys, title="title", xname="step")}
        )
        run.finish()
        ```
        xs can also contain an array of arrays for having different steps for each metric
        <!--yeadoc-test:plot-line-series-double-->
        ```python
        import wandb

        run = wandb.init()
        xs = [[i for i in range(10)], [2 * i for i in range(10)]]
        ys = [[i for i in range(10)], [i**2 for i in range(10)]]
        run.log(
            {"line-series-plot2": wandb.plot.line_series(xs, ys, title="title", xname="step")}
        )
        run.finish()
        ```
    """
    if not isinstance(xs, Iterable):
        raise TypeError(f"Expected xs to be an array instead got {type(xs)}")

    if not isinstance(ys, Iterable):
        raise TypeError(f"Expected ys to be an array instead got {type(xs)}")

    for y in ys:
        if not isinstance(y, Iterable):
            raise TypeError(
                f"Expected ys to be an array of arrays instead got {type(y)}"
            )

    if not isinstance(xs[0], Iterable) or isinstance(xs[0], (str, bytes)):
        xs = [xs for _ in range(len(ys))]
    assert len(xs) == len(ys), "Number of x-lines and y-lines must match"

    if keys is not None:
        assert len(keys) == len(ys), "Number of keys and y-lines must match"

    data = [
        [x, f"key_{i}" if keys is None else keys[i], y]
        for i, (xx, yy) in enumerate(zip(xs, ys))
        for x, y in zip(xx, yy)
    ]

    table = wandb.Table(data=data, columns=["step", "lineKey", "lineVal"])

    return wandb.plot_table(
        "sebastianptz/sample_plot",
        table,
        {"step": "step", "lineKey": "lineKey", "lineVal": "lineVal"},
        {"title": title, "xname": xname or "x"},
    )


def quantile_plot(y, y_hat, quantiles, timestamps=None):
    ## For now assume median is in quantiles and quantiles are symmetric and quantiles are sorted

    color_alpha = np.linspace(1, 0.2, int(np.ceil(len(quantiles)/2)))
    center = int(np.floor(len(quantiles)/2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_hat[:, center], line_width=0, showlegend=False))
    for i in range(center):
        fig.add_trace(go.Scatter(y=y_hat[:, center-i-1], fill='tonexty', fillcolor='rgba(0,80,150,{})'.format(color_alpha[i+1]), line_color='rgba(0,80,150,{})'.format(color_alpha[i+1]/2), name=quantiles[center-i-1]))
    fig.add_trace(go.Scatter(y=y_hat[:, center], line=dict(color='rgba(0,80,150,1)'), name=quantiles[center]))
    for i in range(center):
        fig.add_trace(go.Scatter(y=y_hat[:, center+i+1], fill='tonexty', line_color='rgba(0,80,150,{})'.format(color_alpha[i+1]/2), fillcolor='rgba(0,80,150,{})'.format(color_alpha[i+1]), name=quantiles[center+i+1]))
    
    if y is not None:
        fig.add_trace(go.Scatter(y=y, line=dict(color='black'), name='y'))

    fig.update_layout(
        showlegend=True,
        legend_y=1.07,
    )
    
    if timestamps is not None:
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=np.arange(0, len(y_hat[:, center]), len(y_hat[:, center])//len(timestamps)),
                ticktext=[timestamps[0].strftime("%a,<br>%d %b %y<br>%H:%M")] + [j.strftime("%H:%M") for j in timestamps[1:]],
            ),
                
        )
    return fig


def multiline_plot(data, name):
    n_ts = data.shape[0]
    n_vals = data.shape[1]
    wandb.log(
        {
            f"line_plot_{name}": sample_plot(
                xs=np.arange(1, n_vals + 1),
                ys=[data[i, :] for i in range(n_ts)],
                keys=[str(i) for i in range(n_ts)],
                title=name,
            )
        }
    )


def bar_plot(data, name):
    plot_data = [(str(i), data[i]) for i in range(len(data))]
    table = wandb.Table(data=plot_data, columns=["time_series", name])
    wandb.log(
        {f"bar_plot_{name}": wandb.plot.bar(table, "time_series", name, title=name)}
    )
