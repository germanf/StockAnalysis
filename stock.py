import requests
import json
from datetime import date
import time
from datetime import timedelta
import json


""" The class is used to process and collect stock data """


""" Get price of a stock now """


def GetPrice(s):
    link = "/"
    seq = ("https://api.iextrading.com/1.0/stock", s, "price")
    url = link.join(seq)
    response = requests.get(url)
    data = response.json()
    return data


""" Get period of time when we need to mine for tweets (over 3 days)
    For example, if the input is 2018-5-4, the out put is 2018-5-1,
    2018-5-4, which means from 2018-5-1 to 2018-5-4
"""

def GetPeriodDate(y, m, d):
    datetime = date(y,m,d)
    oneday = timedelta(days=1)
    fourdays = timedelta(days=4)
    start = datetime-fourdays
    end = datetime-oneday
    startDate = str(start.timetuple().tm_year) + "-" + str(start.timetuple().tm_mon) + "-" + str(start.timetuple().tm_mday)
    endDate = str(end.timetuple().tm_year) + "-" + str(end.timetuple().tm_mon) + "-" + str(end.timetuple().tm_mday)
    return startDate, endDate


""" Translate the date into a format that accepted by search_tweets """

def GetCurrentDate(y, m, d):
    return "{}-{:02d}-{}".format(y,m,d)


""" Get the price changed percent on one day """

def GetPriceChangeOnDate(s,date):
    link = "/"
    seq = ("https://api.iextrading.com/1.0/stock", s, "chart/1y")
    url = link.join(seq)
    response = requests.get(url)
    data = response.json()
    for i in data:
        if (i["date"] == date):
            return i["changePercent"]
    return None


""" Get the price changed percent over five days """

def GetPriceChangedOverFiveDays(s, date):
    link = "/"
    seq = ("https://api.iextrading.com/1.0/stock", s, "chart/1m")
    url = link.join(seq)
    response = requests.get(url)
    data = response.json()
    priceChanged = 0
    for i in range(len(data)):
        if (data[i]["date"] == date):
            for j in range(i-1, i-5 if i-5 >= -1 else -1, -1):
                priceChanged+=data[j]["changePercent"]
    return priceChanged


""" Get all stock symbols """

def GetAllStockSymbol():
    url = "https://api.iextrading.com/1.0/tops"
    response = requests.get(url)
    data = response.json()
    for item in data:
        with open ("Stock.txt", "a+") as f:
            json.dump(item, f)
            f.write("\n")


""" Put all symbols in this file into a list """

def GetSymbolsList():
    symbol = []
    with open ("Stock.txt", "r") as f:
        for row in f:
            symbol.append(json.loads(row)["symbol"])
    return symbol


""" Get the price changed over a year (for simple predict machine) """

def GetYearPrice(s):
    link = "/"
    seq = ("https://api.iextrading.com/1.0/stock", s, "chart/1y")
    url = link.join(seq)
    response = requests.get(url)
    data = response.json()
    with open(s+'prices.json', 'w') as outfile:
        json.dump(data, outfile)


""" Prepare the training set for simple predict machine """

def analyze(name):
    prices = json.load(open(name + 'prices.json'))
    for i in range(len(prices)-1):
        f = {}
        changeOverFiveDays = 0
        for j in range(i-1, i-5 if i-5 >= -1 else -1, -1):
            changeOverFiveDays += prices[j]["changePercent"]
        f["changeOverFiveDays"] = changeOverFiveDays
        f["label"] = 1 if prices[i+1]["changePercent"] > 0 else 0
        with open('trainingSet.txt', 'a+') as out:
            json.dump(f, out)
            out.write("\n")


