import dash
from dash import html
from dash import dcc
from dash.dependencies import Output, Input
from dash_extensions import Lottie
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import plotly
import plotly.graph_objs as go
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pyodbc
from PIL import Image  # to load our image
import numpy as np  # to get the color of our image
from io import BytesIO
import base64


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# Lottie urls and option for GIFs ********************************************
url_socials = "https://assets2.lottiefiles.com/packages/lf20_pe4l58xq.json"
url_hashtag = "https://assets10.lottiefiles.com/private_files/lf30_j0plevar.json"
options = dict(loop=True, autoplay=True,
               rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))

# Database credentials for Connecting to DB **********************************
db_connect = pd.read_csv('db_connect.csv')

server = db_connect.secret[0]
database = db_connect.secret[1]
username = db_connect.secret[2]
password = db_connect.secret[3]


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H1('Twitter Sentiment Analysis with ndudataworks'),
                ])
            ], color='rgb(29,161,242)', className='mb-2 text-center'),
            dbc.Card([
                dbc.CardBody([
                    html.H6('***Note: This app collects live data from the\
                        Twitter API, therefore, after clicking "Start\
                            Analysis" loading of data on the app might take\
                                time depending on the engagement on the topic\
                                    you want to analyze.')
                    ])
            ], color="warning", outline=True,)
        ], width=10),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H6('Overall Sentiment')
                ], style={'textAlign': 'center'}),
                dbc.CardBody([
                    html.H2(id='avg-score', children='000'),
                    html.H5(id='sentiment-score', children='000'),
                    dcc.Interval(id='update-ov-score',
                                 disabled=False,
                                 interval=1*5000,
                                 n_intervals=0,
                                 max_intervals=-1)
                ], style={'textAlign': 'center'})
            ], color="info", outline=True)
        ], width=2, className='mt-2'),
    ], className='mb-2 mt-2'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    Lottie(options=options, width="69%", height="69%",
                           url=url_hashtag)
                ])
            ], color="info", outline=True),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4('Enter "track word" and "Enter" to start analysis'),
                    html.Br(),
                    dcc.Input(id='TRACK_WORDS', value='twitter',
                              debounce=True,
                              placeholder='for example: twitter'),
                    dbc.Button("Start Analysis", id="start-button", size="sm",
                               color="info", outline=True, className="me-1",
                               n_clicks=0)
                ])
            ], className='text-center', color="info", outline=True),
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    Lottie(options=options, width="51%", height="51%",
                           url=url_socials)
                ])
            ], color="info", outline=True),
        ], width=2),
    ], className='mb-2'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='scatter-plot', animate=True,
                              config={'displayModeBar': False}),
                    dcc.Interval(id='update-scatter',
                                 disabled=False,
                                 interval=1*5000,
                                 n_intervals=0,
                                 max_intervals=-1)
                ])
            ], color="info", outline=True)
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='bar-chart', animate=True,
                              config={'displayModeBar': False}),
                    dcc.Interval(id='update-bar',
                                 disabled=False,
                                 interval=1*5000,
                                 n_intervals=0,
                                 max_intervals=-1)
                ])
            ], color="info", outline=True)
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Img(id="image_wc", style={'height': '100%',
                                                   'width': '100%'}),
                    dcc.Interval(id='update-wc',
                                 disabled=False,
                                 interval=1*10000,
                                 n_intervals=0,
                                 max_intervals=-1)
                ])
            ], color="info", outline=True)
        ], width=4)
    ], className='mb-2'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='line-mentions', animate=True,
                              config={'displayModeBar': False}),
                    dcc.Interval(id='update-line',
                                 disabled=False,
                                 interval=1*3000,
                                 n_intervals=0,
                                 max_intervals=-1)
                ])
            ], color="info", outline=True)
        ], width=12),
    ], className='mb-2')
], fluid=True)


@app.callback(Output('scatter-plot', 'figure'),
              Input('update-scatter', 'n_intervals'),
              Input('TRACK_WORDS', 'value'))
def update_scatter_plot(num, track_word):
    '''
    updates the scatter plot every 5 seconds

    '''
    if num == 0:
        raise PreventUpdate
    else:
        cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};\
            SERVER='+server+';DATABASE='+database+';UID='+username+';\
                PWD='+password)
        df = pd.read_sql("SELECT TOP 1000 * FROM tweepy WHERE text LIKE\
            '%{}%' ORDER BY created_at DESC;".format(track_word), cnxn)
        # df.sort_values('created_at', inplace=True)

        X = df['polarity']
        Y = df['subjectivity']

        data = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Scatter',
                mode='markers',
                marker=dict(color='rgb(29,161,242)')
                )

        return {'data': [data],
                'layout': go.Layout(xaxis=dict(title='Polarity',
                                               range=[-1, 1]),
                                    yaxis=dict(title='Subjectivity',
                                               range=[0, 1]),
                                    margin=dict(l=50, r=30, t=30, b=50))}

        # fig_scatter = px.scatter(df, x='polarity', y='subjectivity',
        #                          title="Subjectivity vs Polarity")
        # fig_scatter.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        # return fig_scatter


@app.callback(Output('bar-chart', 'figure'),
              Input('update-bar', 'n_intervals'),
              Input('TRACK_WORDS', 'value'))
def update_bar_chart(num, track_word):
    '''
    updates the bar chart every 5 seconds

    '''
    if num == 0:
        raise PreventUpdate
    else:
        cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};\
            SERVER='+server+';DATABASE='+database+';UID='+username+';\
                PWD='+password)
        df = pd.read_sql("SELECT TOP 1000 * FROM tweepy WHERE text LIKE\
            '%{}%' ORDER BY created_at DESC;".format(track_word), cnxn)
        dff = df[["analysis"]].value_counts().head(3)
        dff = dff.to_frame()
        dff.reset_index(inplace=True)
        dff.rename(columns={0: 'Count'}, inplace=True)

        X = dff['analysis']
        Y = dff['Count']

        data = plotly.graph_objs.Bar(
                x=X,
                y=Y,
                name='Bar',
                marker=dict(color='rgb(29,161,242)')
                )

        return {'data': [data],
                'layout': go.Layout(title='Sentiment Score Analysis',
                                    xaxis=dict(title='Sentiment'),
                                    yaxis=dict(title='Count',
                                               range=[0, max(Y)]),
                                    margin=dict(l=50, r=10, t=30, b=50))}


@app.callback(Output('image_wc', 'src'),
              Input('update-wc', 'n_intervals'),
              Input('TRACK_WORDS', 'value'))
def update_wordcloud(num, track_word):
    '''
    updates the wordcloud every 10 seconds

    '''
    if num == 0:
        raise PreventUpdate
    else:
        cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};\
            SERVER='+server+';DATABASE='+database+';UID='+username+';\
                PWD='+password)
        df = pd.read_sql("SELECT TOP 1000 * FROM tweepy WHERE text LIKE\
            '%{}%' ORDER BY created_at DESC;".format(track_word), cnxn)

        content = ' '.join(df["clean_text"])
        content = content.lower()
        stopwords = set(STOPWORDS)

        # Appearance-related
        custom_mask = np.array(Image.open('twitter.png'))
        wc = WordCloud(background_color='white', stopwords=stopwords,
                       max_words=10000,
                       mask=custom_mask, max_font_size=50, contour_width=3,
                       contour_color='rgb(29,161,242)', height=800, width=600)

        wc.generate(content)
        image_colors = ImageColorGenerator(custom_mask)
        wc.recolor(color_func=image_colors)

        fig_wc = wc.to_image()

        img = BytesIO()
        fig_wc.save(img, format='PNG')

        return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())


@app.callback(Output('line-mentions', 'figure'),
              Input('update-line', 'n_intervals'),
              Input('TRACK_WORDS', 'value'))
def update_line_chart(num, track_word):
    '''
    updates the line chart every 10 seconds

    '''
    if num == 0:
        raise PreventUpdate
    else:
        cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};\
            SERVER='+server+';DATABASE='+database+';UID='+username+';\
                PWD='+password)
        df = pd.read_sql("SELECT TOP 100 * FROM tweepy WHERE text LIKE\
            '%{}%' ORDER BY created_at DESC;".format(track_word), cnxn)

        df['created_at'] = pd.to_datetime(df['created_at'])

        result = df[['text']].groupby(df['created_at']).count().reset_index()
        result = result.rename(columns={"text": "Num of '{}' mentions".format(track_word), "created_at": "Time in UTC"})

        X = result['Time in UTC']
        Y = result["Num of '{}' mentions".format(track_word)]

        data = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Scatter',
                mode='lines+markers',
                marker=dict(color='rgb(29,161,242)')
                )

        return {'data': [data],
                'layout': go.Layout(xaxis=dict(title='Time in UTC',
                                               range=[min(X), max(X)]),
                                    yaxis=dict(title="Number of mentions",
                                               range=[min(Y), max(Y)]),
                                    margin=dict(l=50, r=30, t=30, b=50))}


@app.callback(Output('avg-score', 'children'),
              Output('sentiment-score', 'children'),
              Input('update-ov-score', 'n_intervals'),
              Input('TRACK_WORDS', 'value'))
def update_overall_sentiment(num, track_word):
    '''
    updates the scatter plot every 5 seconds

    '''
    if num == 0:
        raise PreventUpdate
    else:
        cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};\
            SERVER='+server+';DATABASE='+database+';UID='+username+';\
                PWD='+password)
        df = pd.read_sql("SELECT TOP 1000 * FROM tweepy WHERE text LIKE\
            '%{}%' ORDER BY created_at DESC;".format(track_word), cnxn)

        score = df['polarity'].mean()

        if score < 0:
            avg_score = round(score, 2)
            avg_sentiment = 'Negative'
        elif score == 0:
            avg_score = round(score, 2)
            avg_sentiment = "Neutral"
        else:
            avg_score = round(score, 2)
            avg_sentiment = "Positive"

    return avg_score, avg_sentiment


if __name__ == '__main__':
    app.run_server(debug=True)
