import twitter
import json
import sys
import time
from urllib2 import URLError
from httplib import BadStatusLine
from prettytable import PrettyTable
from yahoo_finance import Share
import requests
import numpy as np
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize
import nltk
import codecs
import stock
import predict
import math
import plot
from datetime import date


from nltk.sentiment.vader import SentimentIntensityAnalyzer
"""
Login function (from cookbook)

"""
def oauth_login():
    CONSUMER_KEY = '09RpgWjQRId9OTusljygc2Wmv'
    CONSUMER_SECRET = 'cWx4l6oD6M3n4pX1ARurQfPbRzV9jYunhF4pu6fpu5EcHJtizy'
    OAUTH_TOKEN='958418693083549696-BYLrMCW32guPZ5SqqXenQem5cP5jGQK'
    OAUTH_TOKEN_SECRET='5jSmekrWQ2JmzqXhhVRPmjEkM1q2N2YqR12ySU3qPcIGQ'
    auth = twitter.OAuth(OAUTH_TOKEN,OAUTH_TOKEN_SECRET,CONSUMER_KEY,CONSUMER_SECRET)
    twitter_api = twitter.Twitter(auth = auth)
    return twitter_api

"""
Used to make twitter request (from cookbook)

"""
def make_twitter_request(twitter_api_func, max_errors=10, *args, **kw):
    # A nested helper function that handles common HTTPErrors. Return an updated
    # value for wait_period if the problem is a 500 level error. Block until the
    # rate limit is reset if it's a rate limiting issue (429 error). Returns None
    # for 401 and 404 errors, which requires special handling by the caller.
    def handle_twitter_http_error(e, wait_period=2, sleep_when_rate_limited=True):

        if wait_period > 3600:  # Seconds
            print >> sys.stderr, 'Too many retries. Quitting.'
            raise e

        # See https://dev.twitter.com/docs/error-codes-responses for common codes

        if e.e.code == 401:
            print >> sys.stderr, 'Encountered 401 Error (Not Authorized)'
            return None
        elif e.e.code == 404:
            print >> sys.stderr, 'Encountered 404 Error (Not Found)'
            return None
        elif e.e.code == 429:
            print >> sys.stderr, 'Encountered 429 Error (Rate Limit Exceeded)'
            if sleep_when_rate_limited:
                print >> sys.stderr, "Retrying in 15 minutes...ZzZ..."
                sys.stderr.flush()
                time.sleep(60 * 15 + 5)
                print >> sys.stderr, '...ZzZ...Awake now and trying again.'
                return 2
            else:
                raise e  # Caller must handle the rate limiting issue
        elif e.e.code in (500, 502, 503, 504):
            print >> sys.stderr, 'Encountered %i Error. Retrying in %i seconds' % (e.e.code, wait_period)
            time.sleep(wait_period)
            wait_period *= 1.5
            return wait_period
        else:
            raise e

    # End of nested helper function

    wait_period = 2
    error_count = 0

    while True:
        try:
            return twitter_api_func(*args, **kw)
        except twitter.api.TwitterHTTPError, e:
            error_count = 0
            wait_period = handle_twitter_http_error(e, wait_period)
            if wait_period is None:
                return
        except URLError, e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print >> sys.stderr, "URLError encountered. Continuing."
            if error_count > max_errors:
                print >> sys.stderr, "Too many consecutive errors...bailing out."
                raise
        except BadStatusLine, e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print >> sys.stderr, "BadStatusLine encountered. Continuing."
            if error_count > max_errors:
                print >> sys.stderr, "Too many consecutive errors...bailing out."
                raise




def twitter_search(twitter_api, q, max_results=200, **kw):
    # See https://dev.twitter.com/docs/api/1.1/get/search/tweets and
    # https://dev.twitter.com/docs/using-search for details on advanced
    # search criteria that may be useful for keyword arguments

    # See https://dev.twitter.com/docs/api/1.1/get/search/tweets
    search_results = twitter_api.search.tweets(q=q, count=100, **kw)

    statuses = search_results['statuses']

    # Iterate through batches of results by following the cursor until we
    # reach the desired number of results, keeping in mind that OAuth users
    # can "only" make 180 search queries per 15-minute interval. See
    # https://dev.twitter.com/docs/rate-limiting/1.1/limits
    # for details. A reasonable number of results is ~1000, although
    # that number of results may not exist for all queries.

    # Enforce a reasonable limit
    max_results = min(1000, max_results)

    for _ in range(10):  # 10*100 = 1000
        try:
            next_results = search_results['search_metadata']['next_results']
        except KeyError, e:  # No more results when next_results doesn't exist
            break

        # Create a dictionary from next_results, which has the following form:
        # ?max_id=313519052523986943&q=NCAA&include_entities=1
        kwargs = dict([kv.split('=')
                       for kv in next_results[1:].split("&")])

        search_results = twitter_api.search.tweets(**kwargs)
        statuses += search_results['statuses']

        if len(statuses) > max_results:
            break

    return statuses





""" Search the tweets that contain our stock symbol, and write the data mined to a file """


def search_stock_tweets(twitter_api, searched, max_results, since, until):
    r = twitter_search(twitter_api, "$" + searched, max_results= max_results, result_type = 'mixed', lang = 'en', since= since, until= until, tweet_mode="extended" )
    with open(searched + ".txt", 'w+') as f:
        for x in r:
            if ('full_text' in x.keys()):

                f.write("text: " + x['full_text'].encode('utf-8') + "\n")
            else:
                f.write("text: " + x['text'].encode('utf-8') + "\n")
            f.write("time: " + x['created_at'].encode('utf-8') + "\n")



""" Analyze the stock symbol by collecting the features we need and put them into our machine """


def analyze(searched, date):

    """ Download what we need """
    try:
        tokenize.sent_tokenize("")
    except:
        nltk.download('punkt')

    try:
        sid = SentimentIntensityAnalyzer()
    except:
        nltk.download('vader_lexicon')
        sid = SentimentIntensityAnalyzer()

    sentence_list = []
    """ Open the file that stores the tweets we mined """
    f = codecs.open(searched+'.txt', encoding='utf-8')
    for line in f:
        if(line.startswith("text:")):
            sentence_list.extend(tokenize.sent_tokenize(line))
    f.close()
    pos = 0
    pos_count = 0
    neg = 0
    neg_count = 0
    neu_count = 0
    count = 0

    """ Analyze sentiment """
    for sentence in sentence_list:
        count += 1
        ss = sid.polarity_scores(sentence)
        if (ss['compound'] > 0.0):
            pos += ss['compound']
            pos_count+=1
        if (ss['compound'] < 0.0):
            neg += ss['compound']
            neg_count+=1
        if (ss['compound'] == 0.0):
            neu_count += 1

    """ Calculate the ratio """
    ratio = pos/(neg if neg != 0 else 0.001) if pos > neg else (neg/(pos if pos != 0 else 0.001))*-1

    """ 
    Process the data that we have and transform them into features
        We have three features: 
        1. ratio of positive and negative tweets, 
        2. count of tweets that are talking about this stock,
        3. probability of going up predicted by "simple predict" machine, which is
        predicting only based on the history stock price data. 
    """
    trainFeature = {}
    trainFeature["ratio"] = math.fabs(ratio)
    trainFeature["count"] = count
    change = stock.GetPriceChangeOnDate(searched, date)

    """ 1 means go up, 0 means go down """
    trainFeature["label"] = 1 if change > 0 else 0
    priceChangedOverFiveDays = stock.GetPriceChangedOverFiveDays(searched, date)

    """ Get prediction result from "simple predict" machine """
    simple_predict_result = predict.SimplePredict("simplePredict.sav", priceChangedOverFiveDays) - 0.5
    trainFeature["simplePredictResult"] = simple_predict_result

    """ Input to our data set """
    with open('trainingSetWithSentiment2.txt', 'a+') as out:
        json.dump(trainFeature, out)
        out.write("\n")


""" Similar to function analyze, but we remove some lines of code for easy to demo """

def predictStock(searched, date):

    """ Download what we need """
    try:
        tokenize.sent_tokenize("")
    except:
        nltk.download('punkt')
    try:
        sid = SentimentIntensityAnalyzer()
    except:
        nltk.download('vader_lexicon')
        sid = SentimentIntensityAnalyzer()
    sentence_list = []

    """ Open the file that stores the tweets we mined """
    f= codecs.open(searched+'.txt', encoding='utf-8')
    for line in f:
        if(line.startswith("text:")):
            sentence_list.extend(tokenize.sent_tokenize(line))
    f.close()

    pos = 0
    pos_count = 0
    neg = 0
    neg_count = 0
    neu_count = 0
    count = 0

    """ Analyze sentiment """
    for sentence in sentence_list:
        count += 1
        ss = sid.polarity_scores(sentence)
        if (ss['compound'] > 0.0):
            pos += ss['compound']
            pos_count+=1
        if (ss['compound'] < 0.0):
            neg += ss['compound']
            neg_count+=1
        if (ss['compound'] == 0.0):
            neu_count += 1

    """ Calculate the ratio """
    ratio = pos/(neg if neg != 0 else 0.001) if pos > neg else (neg/(pos if pos != 0 else 0.001))*-1
    plot.pie_pos_neg(pos, neg * -1)
    plot.pie_pos_neg_neu(pos_count, neg_count, neu_count)

    """ 
    Process the data that we have and transform them into features
        We have three features: 
        1. ratio of positive and negative tweets, 
        2. count of tweets that are talking about this stock,
        3. probability of going up predicted by "simple predict" machine, which is
        predicting only based on the history stock price data. 
    """
    trainFeature = {}
    trainFeature["ratio"] = math.fabs(ratio)
    trainFeature["count"] = count
    priceChangedOverFiveDays = stock.GetPriceChangedOverFiveDays(searched, date)
    simple_predict_result = predict.SimplePredict("simplePredict.sav", priceChangedOverFiveDays) - 0.5
    trainFeature["simplePredictResult"] = simple_predict_result
    with open('trainingSetFromUserInput.txt', 'a+') as out:
        json.dump(trainFeature, out)
        out.write("\n")
    inputTuple = (trainFeature['count'], trainFeature['ratio'], trainFeature['simplePredictResult'])
    predict.RandomForestPredict(inputTuple)
    predict.LogisticRegressionPredict(inputTuple)


""" Train function that we used to train our machine"""
def train(symbol):
    search_stock_tweets(twitter_api, symbol, 1000, "2018-5-1", "2018-5-3")
    analyze(symbol, "2018-05-04")


"""Predict the stock price today just for easy to demo our project"""
def predictStockToday(symbol):
    currentDay = date.today()
    year = currentDay.timetuple().tm_year
    month = currentDay.timetuple().tm_mon
    day = currentDay.timetuple().tm_mday
    start, end = stock.GetPeriodDate(year, month, day)
    current = stock.GetCurrentDate(year, month, day)
    search_stock_tweets(twitter_api=twitter_api, searched=symbol, max_results=1000, since=start, until=end)
    predictStock(symbol, current)



if __name__ == "__main__":

    """ Login twitter api"""
    twitter_api = oauth_login()

    """Get All Symbols from IEX_API"""
    symbols = stock.GetSymbolsList()

    """ Instruction: Input the symbol of stock you want to predict (e.g.: MSFT) 
        Since the accuracy of RandomForest is higher, when Random forest algorithm
        predict the stock will go up or go down but the probability predicted by 
        logistic regression is around 50, then please follow the prediction made by 
        Random forest algorithm
    """
    user_input = raw_input("Please input a symbol that you want to predict for tomorrow")

    if (user_input in symbols):
        predictStockToday(user_input)
    else :
        print ("The symbol you typed is not available")
