import requests
import json
from datetime import date
import time
from datetime import timedelta
import json

def GetPrice(s):
    link = "/"
    seq = ("https://api.iextrading.com/1.0/stock", s, "price")
    url = link.join(seq)
    response = requests.get(url)
    data = response.json()
    return data

def GetPeriodDate(y, m, d):
    datetime = date(y,m,d)
    oneday = timedelta(days=1)
    fourdays = timedelta(days=4)
    start = datetime-fourdays
    end = datetime-oneday
    startDate = str(start.timetuple().tm_year) + "-" + str(start.timetuple().tm_mon) + "-" + str(start.timetuple().tm_mday)
    endDate = str(end.timetuple().tm_year) + "-" + str(end.timetuple().tm_mon) + "-" + str(end.timetuple().tm_mday)
    return startDate, endDate

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

def GetAllStockSymbol():
    url = "https://api.iextrading.com/1.0/tops"
    response = requests.get(url)
    data = response.json()
    for item in data:
        with open ("Stock.txt", "a+") as f:
            json.dump(item, f)
            f.write("\n")

def GetSymbolsList():
    symbol = []
    with open ("Stock.txt", "r") as f:
        for row in f:
            symbol.append(json.loads(row)["symbol"])
    return symbol


def GetYearPrice(s):
    link = "/"
    seq = ("https://api.iextrading.com/1.0/stock", s, "chart/1y")
    url = link.join(seq)
    response = requests.get(url)
    data = response.json()
    with open(s+'prices.json', 'w') as outfile:
        json.dump(data, outfile)

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


