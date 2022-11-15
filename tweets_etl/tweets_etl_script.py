import attributes
import pandas as pd
import pyodbc
import tweepy
from textblob import TextBlob
import re
import time
from datetime import datetime
# Streaming With Tweepy
# http://docs.tweepy.org/en/v3.4.0/streaming_how_to.html#streaming-with-tweepy

timestart = datetime.now()


# Subjectivity and Polarity and Analysis functions ***************************
def clean_tweet(text):
    """
    This function is for cleaning up the Tweet
    ingested using the twitter API (tweepy)

    Args:
        text (string): Tweets from the Twitter API
    """

    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Removes @mentions
    text = re.sub(r'#', '', text)  # Removes hashtags '#' symbol
    text = re.sub(r'RT[\s]+', '', text)  # Removes RT (for retweet)
    text = re.sub(r'https?:\/\/\S+', '', text)  # Removes hyperlink

    return text


def deEmojify(text):
    '''
    Strip all non-ASCII characters to remove emoji characters
    '''
    if text:
        return text.encode('ascii', 'ignore').decode('ascii')
    else:
        return None


def get_analysis(score):
    """
    This function will evaluate whether the polarity
    score is negative, neutral and positive

    Args:
        score ([float]): [This is the polarity score from sentiment]

    """

    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"


# Database credentials + Connecting to DB ************************************
db_connect = pd.read_csv('db_connect.csv')

server = db_connect.secret[0]
database = db_connect.secret[1]
username = db_connect.secret[2]
password = db_connect.secret[3]

cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password)


def create_sql_tbl(cnxn):
    mycursor = cnxn.cursor()
    mycursor.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = '{0}'
            """.format(attributes.TABLE_NAME))
    if mycursor.fetchone()[0] != 1:
        mycursor.execute("CREATE TABLE {} ({})".format(attributes.TABLE_NAME,
                                                       attributes.TABLE_ATTRIBUTES))
        cnxn.commit()
    mycursor.close()


create_sql_tbl(cnxn)


# Defining Tweepy MyStreamlinster ********************************************
class MyStreamListener(tweepy.Stream):
    '''
    Twitter identifies tweets as “status updates”. So the Status class in
    tweepy has properties describing the tweet.
    https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object.html
    '''

    def on_status(self, status):
        '''
        Extract info from tweets
        '''

        if status.retweeted:
            # Avoid retweeted info, and only original tweets will be received
            return True
        # Extract attributes from each tweet
        id_str = status.id_str
        created_at = status.created_at
        text = status.text
        clean_text = deEmojify(status.text)    # Pre-processing the text
        clean_text = clean_tweet(status.text)    # Pre-processing the text
        polarity = TextBlob(clean_text).sentiment.polarity
        subjectivity = TextBlob(clean_text).sentiment.subjectivity
        analysis = get_analysis(polarity)
        user_created_at = status.user.created_at
        user_location = deEmojify(status.user.location)

        print(status.text)

        # Store all data in SQL Server
        mycursor = cnxn.cursor()
        sql = """INSERT INTO {} (id_str, created_at, text, clean_text, polarity, subjectivity, analysis, user_created_at, user_location) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""".format(attributes.TABLE_NAME)
        val = (id_str, created_at, text, clean_text, polarity, subjectivity, analysis, user_created_at, user_location)
        mycursor.execute(sql, val)
        cnxn.commit()
        mycursor.close()

    def on_request_error(self, status_code):
        '''
        Since Twitter API has rate limits, stop srcraping data as it exceed to the thresold.
        '''
        if status_code == 420:
            # return False to disconnect the stream
            return False


credentials = pd.read_csv('tweepyKeys.csv')

while True:

    try:
        myStreamListener = MyStreamListener(consumer_key=credentials['Keys'][0],
                                            consumer_secret=credentials['Keys'][1],
                                            access_token=credentials['Keys'][2],
                                            access_token_secret=credentials['Keys'][3])
        myStreamListener.filter(languages=["en"], track=attributes.TRACK_WORDS)
        time.sleep(60)
        myStreamListener.disconnect()
    except Exception as e:
        print(str(e))
        time.sleep(5)
