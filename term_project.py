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

"""
Used to get user profile json (from cookbook)

"""
def get_user_profile(twitter_api, screen_names=None, user_ids=None):
    # Must have either screen_name or user_id (logical xor)
    assert (screen_names != None) != (user_ids != None), "Must have screen_names or user_ids, but not both"

    items_to_info = {}

    items = screen_names or user_ids

    while len(items) > 0:

        # Process 100 items at a time per the API specifications for /users/lookup.
        # See https://dev.twitter.com/docs/api/1.1/get/users/lookup for details.

        items_str = ','.join([str(item) for item in items[:100]])
        items = items[100:]

        if screen_names:
            response = make_twitter_request(twitter_api.users.lookup,
                                            screen_name=items_str)
        else:  # user_ids
            response = make_twitter_request(twitter_api.users.lookup,
                                            user_id=items_str)
        #if (response == None):
        #    continue

        for user_info in response:
            if screen_names:
                items_to_info[user_info['screen_name']] = user_info
            else:  # user_ids
                items_to_info[user_info['id']] = user_info

    return items_to_info
from collections import Counter

def get_common_tweet_entities(statuses, entity_threshold=3):

    # Create a flat list of all tweet entities
    tweet_entities = [  e
                        for status in statuses
                            for entity_type in extract_tweet_entities([status])
                                for e in entity_type
                     ]

    c = Counter(tweet_entities).most_common()

    # Compute frequencies
    return [ (k,v)
             for (k,v) in c
                 if v >= entity_threshold
           ]


def extract_tweet_entities(statuses):
    # See https://dev.twitter.com/docs/tweet-entities for more details on tweet
    # entities

    if len(statuses) == 0:
        return [], [], [], [], []

    screen_names = [user_mention['screen_name']
                    for status in statuses
                    for user_mention in status['entities']['user_mentions']]

    hashtags = [hashtag['text']
                for status in statuses
                for hashtag in status['entities']['hashtags']]

    urls = [url['expanded_url']
            for status in statuses
            for url in status['entities']['urls']]

    symbols = [symbol['text']
               for status in statuses
               for symbol in status['entities']['symbols']]

    # In some circumstances (such as search results), the media entity
    # may not appear
    if 'media' in status['entities'].keys():
        media = [media['url']
                 for status in statuses
                 for media in status['entities']['media']]
    else:
        media = []

    return screen_names, hashtags, urls, media, symbols



from functools import partial
from sys import maxint
def get_friends_followers_ids(twitter_api, screen_name=None, user_id=None,
                              friends_limit=maxint, followers_limit=maxint):
    # Must have either screen_name or user_id (logical xor)
    assert (screen_name != None) != (user_id != None), "Must have screen_name or user_id, but not both"

    # See https://dev.twitter.com/docs/api/1.1/get/friends/ids and
    # https://dev.twitter.com/docs/api/1.1/get/followers/ids for details
    # on API parameters

    get_friends_ids = partial(make_twitter_request, twitter_api.friends.ids,
                              count=5000)
    get_followers_ids = partial(make_twitter_request, twitter_api.followers.ids,
                                count=5000)

    friends_ids, followers_ids = [], []

    for twitter_api_func, limit, ids, label in [
        [get_friends_ids, friends_limit, friends_ids, "friends"],
        [get_followers_ids, followers_limit, followers_ids, "followers"]
    ]:

        if limit == 0: continue

        cursor = -1
        while cursor != 0:

            # Use make_twitter_request via the partially bound callable...
            if screen_name:
                response = twitter_api_func(screen_name=screen_name, cursor=cursor)
            else:  # user_id
                response = twitter_api_func(user_id=user_id, cursor=cursor)

            if response is not None:
                ids += response['ids']
                cursor = response['next_cursor']

            print >> sys.stderr, 'Fetched {0} total {1} ids for {2}'.format(len(ids),
                                                                            label, (user_id or screen_name))

            # XXX: You may want to store data during each iteration to provide an
            # an additional layer of protection from exceptional circumstances

            if len(ids) >= limit or response is None:
                break

    # Do something useful with the IDs, like store them to disk...
    return friends_ids[:friends_limit], followers_ids[:followers_limit]


def analyze_tweet_content(statuses):
    if len(statuses) == 0:
        print "No statuses to analyze"
        return

    # A nested helper function for computing lexical diversity
    def lexical_diversity(tokens):
        return 1.0 * len(set(tokens)) / len(tokens)

        # A nested helper function for computing the average number of words per tweet

    def average_words(statuses):
        total_words = sum([len(s.split()) for s in statuses])
        return 1.0 * total_words / len(statuses)

    status_texts = [status['text'] for status in statuses]
    screen_names, hashtags, urls, media, _ = extract_tweet_entities(statuses)

    # Compute a collection of all words from all tweets
    words = [w
             for t in status_texts
             for w in t.split()]

    print "Lexical diversity (words):", lexical_diversity(words)
    print "Lexical diversity (screen names):", lexical_diversity(screen_names)
    print "Lexical diversity (hashtags):", lexical_diversity(hashtags)
    print "Averge words per tweet:", average_words(status_texts)

def analyze_favorites(twitter_api, screen_name, entity_threshold=2):
    # Could fetch more than 200 by walking the cursor as shown in other
    # recipes, but 200 is a good sample to work with.
    favs = twitter_api.favorites.list(screen_name=screen_name, count=200)
    print "Number of favorites:", len(favs)

    # Figure out what some of the common entities are, if any, in the content

    common_entities = get_common_tweet_entities(favs,
                                                entity_threshold=entity_threshold)

    # Use PrettyTable to create a nice tabular display

    pt = PrettyTable(field_names=['Entity', 'Count'])
    [pt.add_row(kv) for kv in common_entities]
    pt.align['Entity'], pt.align['Count'] = 'l', 'r'  # Set column alignment

    print
    print "Common entities in favorites..."
    print pt

    # Print out some other stats
    print
    print "Some statistics about the content of the favorities..."
    print
    analyze_tweet_content(favs)


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


def harvest_user_timeline(twitter_api, screen_name=None, user_id=None, max_results=1000):
    assert (screen_name != None) != (user_id != None), "Must have screen_name or user_id, but not both"

    kw = {  # Keyword args for the Twitter API call
        'count': 200,
        'trim_user': 'true',
        'include_rts': 'true',
        'since_id': 1
    }

    if screen_name:
        kw['screen_name'] = screen_name
    else:
        kw['user_id'] = user_id

    max_pages = 16
    results = []

    tweets = make_twitter_request(twitter_api.statuses.user_timeline, **kw)

    if tweets is None:  # 401 (Not Authorized) - Need to bail out on loop entry
        tweets = []

    results += tweets

    print >> sys.stderr, 'Fetched %i tweets' % len(tweets)

    page_num = 1

    # Many Twitter accounts have fewer than 200 tweets so you don't want to enter
    # the loop and waste a precious request if max_results = 200.

    # Note: Analogous optimizations could be applied inside the loop to try and
    # save requests. e.g. Don't make a third request if you have 287 tweets out of
    # a possible 400 tweets after your second request. Twitter does do some
    # post-filtering on censored and deleted tweets out of batches of 'count', though,
    # so you can't strictly check for the number of results being 200. You might get
    # back 198, for example, and still have many more tweets to go. If you have the
    # total number of tweets for an account (by GET /users/lookup/), then you could
    # simply use this value as a guide.

    if max_results == kw['count']:
        page_num = max_pages  # Prevent loop entry

    while page_num < max_pages and len(tweets) > 0 and len(results) < max_results:
        # Necessary for traversing the timeline in Twitter's v1.1 API:
        # get the next query's max-id parameter to pass in.
        # See https://dev.twitter.com/docs/working-with-timelines.
        kw['max_id'] = min([tweet['id'] for tweet in tweets]) - 1

        tweets = make_twitter_request(twitter_api.statuses.user_timeline, **kw)
        results += tweets

        print >> sys.stderr, 'Fetched %i tweets' % (len(tweets),)

        page_num += 1

    print >> sys.stderr, 'Done fetching tweets'

    return results[:max_results]

twitter_api = oauth_login()
#screen_name = "EAFay23"
#profile = get_user_profile(twitter_api, [screen_name])
#print (json.dumps(profile, indent = 1))
#analyze_favorites(twitter_api, screen_name)
"""
url = ("https://api.iextrading.com/1.0/stock/aapl/previous")
rep = requests.get(url)
sdata = rep.json()
print (json.dumps(sdata,indent = 1))

"""
def havest_user_stock_tweet(stock_name,user_name,result_limit, key_word):
	time_of_tweet=""
	stock_tweets = harvest_user_timeline(twitter_api, screen_name=user_name,max_results=result_limit,)
	### open file and test output
	for x in stock_tweets:
		if ('full_text' in x.keys()):
			if(key_word in x['full_text']):
				time=x['created_at']
				time_of_tweet=time[4:11]+time[len-4:len]
				with open(time_of_tweet + ".txt", 'a+') as k:
				    k.write("text: " + x['full_text'].encode('utf-8') + "\n")
		else:
			if(key_word in x['text']):
				time = x['created_at']
				time_of_tweet = time[4:11] + time[len(time) - 4:len(time)]
				with open(time_of_tweet + ".txt", 'a+') as k:
				    k.write("text: " + x['text'].encode('utf-8') + "\n")
"""
user_name="HIDEO_KOJIMA_EN"
result_limit=200
key_word="DEATH STRANDING"
havest_user_stock_tweet(key_word,user_name,result_limit)
"""
def search_stock_tweets(twitter_api, searched, max_results, since, until):
    r = twitter_search(twitter_api, "$" + searched, max_results= max_results, result_type = 'mixed', lang = 'en', since= since, until= until, tweet_mode="extended" )
    with open(searched + ".txt", 'w+') as f:
        for x in r:
            if ('full_text' in x.keys()):

                f.write("text: " + x['full_text'].encode('utf-8') + "\n")
            else:
                f.write("text: " + x['text'].encode('utf-8') + "\n")
            f.write("time: " + x['created_at'].encode('utf-8') + "\n")



### open file and out put.

"""

"""
        


"""


history = harvest_user_timeline(twitter_api, screen_name= "jimcramer", max_results= 200)
print (json.dumps(history, indent = 1))
"""
"""
nltk.download('subjectivity')
n_instances = 100
subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
train_subj_docs = subj_docs[:80]
test_subj_docs = subj_docs[80:100]
train_obj_docs = obj_docs[:80]
test_obj_docs = obj_docs[80:100]
training_docs = train_subj_docs+train_obj_docs
testing_docs = test_subj_docs+test_obj_docs
sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
"""

def analyze(searched, date):
    nltk.download('punkt')

    try:
        sid = SentimentIntensityAnalyzer()
    except:
        nltk.download('vader_lexicon')

        sid = SentimentIntensityAnalyzer()
    sentence_list = []
    f= codecs.open(searched+'.txt', encoding='utf-8')
    for line in f:
        if(line.startswith("text:")):
            sentence_list.extend(tokenize.sent_tokenize(line))
    f.close()
    pos = 0
    neg = 0
    count = 0
    for sentence in sentence_list:
        count += 1
        ss = sid.polarity_scores(sentence)
        if (ss['compound'] > 0.0):
            pos += ss['compound']
        if (ss['compound'] < 0.0):
            neg += ss['compound']
        print (ss)
    print ("pos: ", pos)
    print ("neg: ", neg)
    ratio = pos/(neg if neg != 0 else 0.001) if pos > neg else (neg/(pos if pos != 0 else 0.001))*-1
    trainFeature = {}
    trainFeature["ratio"] = math.fabs(ratio)
    trainFeature["count"] = count
    change = stock.GetPriceChangeOnDate(searched, date)
    print ("change: ", change)
    trainFeature["label"] = 1 if change > 0 else 0
    priceChangedOverFiveDays = stock.GetPriceChangedOverFiveDays(searched, date)
    simple_predict_result = predict.SimplePredict("simplePredict.sav", priceChangedOverFiveDays) - 0.5
    trainFeature["simplePredictResult"] = simple_predict_result

    with open('trainingSetWithSentiment2.txt', 'a+') as out:
        json.dump(trainFeature, out)
        out.write("\n")


def train (symbol):
    search_stock_tweets(twitter_api, symbol, 1000, "2018-4-23", "2018-4-26")
    analyze( symbol, "2018-04-27")

symbolsList = stock.GetSymbolsList()[:50]
print(symbolsList)
for i in symbolsList:
    train(i)

#for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
#    print('{0}: {1}'.format(key, value))
#print (json.dumps(r, indent = 1))
"""
fig = plt.figure()
plt.show()
"""
'''
input: a user name
output: text\n time\n
'''