import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter, Figure, Layout
import datetime


plotly.tools.set_credentials_file(username='jiaxunsong', api_key='pTQyltRjsA0o0bOz3YkM')


# Plot a pie chart with pos% and neg%
def pie_pos_neg(num_of_pos, num_of_neg):
    pos_neg_labels = ['Positive', 'Negative']
    pos_neg_values = [num_of_pos, num_of_neg]

    trace = go.Pie(labels=pos_neg_labels, values=pos_neg_values)

    plot([trace])


# Plot a pie chart with #of pos, #of neg and #of neu
def pie_pos_neg_neu(num_of_pos, num_of_neg, num_of_neu):
    pos_neg_neu_labels = ['Positive', 'Negative', 'Neutral']
    pos_neg_neu_values = [num_of_pos, num_of_neg, num_of_neu]
    plot([go.Pie(labels=pos_neg_neu_labels, values=pos_neg_neu_values)])


# Using time series to plot stock prices
# Stock_price is a list of 6-day stock price (double)
# comments is a list of compounds (double)
def stock_vs_sentiment(stock_price, comments):
    x = [datetime.datetime(year=2018, month=4, day=22),
         datetime.datetime(year=2018, month=4, day=23),
         datetime.datetime(year=2018, month=4, day=24),
         datetime.datetime(year=2018, month=4, day=25),
         datetime.datetime(year=2018, month=4, day=26),
         datetime.datetime(year=2018, month=4, day=27)]

    # Plot a bar chart and a line chart in a same graph
    # Replace y values by actual values, replace names by data names
    bar_chart = go.Bar(x=x,
                       y=comments, name='yaxis_data', yaxis='y1')

    line_chart = go.Scatter(x=x, y=stock_price, name='stock price', yaxis='y2')
    layout = go.Layout(title = 'something', yaxis=dict(title='yaxis title'),
                       yaxis2=dict(title='stock price', overlaying='y', side='right'))

    combine_data = go.Figure(data=[bar_chart, line_chart], layout = layout)
    plot(combine_data, filename='combine')


# examples
pie_pos_neg(22,7)
#pie_pos_neg_neu(20,7,12)

stock_price = [100,300,600,400,200,400]
comments = [1,3,6,4,2,4]
stock_vs_sentiment(stock_price, comments)
