import numpy as np
import plotly.graph_objects as go

def plot_line(x, y, name, filename=None):

    trace1 = go.Bar(x=x, y=y[0], texttemplate="%{y}", textposition="outside", name='budget', marker_color='indianred')
    trace2 = go.Bar(x=x, y=y[1], texttemplate="%{y}", textposition="outside", name='comptes', marker_color='lightsalmon')
    layout = go.Layout(
        title = name,
        titlefont = dict(size=36),
        barmode = 'group',
        bargap=0.35,
        bargroupgap=0.1,
        # xaxis = dict(title='juillet à juin'),
        yaxis = dict(title='CHF'),
    )

    fig = go.Figure(data = [trace1, trace2], layout = layout)

    if filename!=None:
        fig.write_html(filename)
    else:
        fig.show()

def plot_combine(x, y, name, filename=None):

    trace1 = go.Bar(x=x[0], y=y[0], name=name[0])
    trace2 = go.Bar(x=x[1], y=y[1], name=name[1])
    trace3 = go.Bar(x=x[2], y=y[2], name=name[2])
    layout = go.Layout(
        title = name,
        titlefont = dict(size=36),
        barmode = 'stack',
        bargap=0.35,
        bargroupgap=0.1,
        # xaxis = dict(title='juillet à juin'),
        yaxis = dict(title='CHF'),
    )

    fig = go.Figure(data = [trace1, trace2, trace3], layout = layout)

    if filename!=None:
        fig.write_html(filename)
    else:
        fig.show()

def plot_year(labels, values, name, filename=None):
    trace = go.Pie(labels, values)
    layout = go.Layout(
        title = name,
        titlefont = dict(size=36),
    )

    fig = go.Figure(data=[trace], layout=layout)

    if filename!=None:
        fig.write_html(filename)
    else:
        fig.show()

