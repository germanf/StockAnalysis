from sklearn import linear_model, decomposition
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


""" This class is used to train machine and make prediction """


""" Train "simple predict" machine, which is predicting only based on history stock data"""

def SimpleTrain():
    features = []
    labels = []
    with open("trainingSet.txt") as set:
        for feature in set:
            FLDict = json.loads(feature)
            FT = (FLDict["changeOverFiveDays"], 1)
            LT = FLDict["label"]
            features.append(FT)
            labels.append(LT)
    pca = linear_model.LogisticRegression()
    pca.fit(features, labels)

    filename = 'simplePredict.sav'
    pickle.dump(pca, open(filename, 'wb'))


""" Prediction made by "simple predict" machine"""

def SimplePredict(filename, changeOverTime):
    trained = pickle.load(open(filename, 'rb'))
    probability = trained.predict_proba([(changeOverTime, 1)])
    return probability[0][1]


""" Train machine based on logistic regression algorithm"""

def SentimentTrain(until):
    features = []
    labels = []
    count = 0
    with open("trainingSetWithSentiment2.txt") as set:
        for feature in set:
            if (count > until):
                break
            count +=1
            FLDict = json.loads(feature)
            FT = (FLDict["count"], FLDict["ratio"], FLDict["simplePredictResult"])
            LT = FLDict["label"]
            features.append(FT)
            labels.append(LT)
    pca = linear_model.LogisticRegression()
    pca.fit(features, labels)

    filename = 'SentimentPredict3.sav'

    """ Save our machine by using pickle """
    pickle.dump(pca, open(filename, 'wb'))


""" Train machine based on random forest algorithm"""

def SentimentRandomForestTrain(until):
    features = []
    labels = []
    count = 0
    with open("trainingSetWithSentiment2.txt") as set:
        for feature in set:
            if (count > until):
                break
            count += 1
            FLDict = json.loads(feature)
            FT = (FLDict["count"], FLDict["ratio"], FLDict["simplePredictResult"])
            LT = FLDict["label"]
            features.append(FT)
            labels.append(LT)
    pca = RandomForestClassifier()
    pca.fit(features, labels)

    """ Save our machine by using picke """
    filename = 'SentimentRandomForestPredict5.sav'
    pickle.dump(pca, open(filename, 'wb'))


""" Get prediction"""
def Predict(filename, inputTuple):
    trained = pickle.load(open(filename, 'rb'))
    probability = trained.predict_proba([inputTuple])
    print ("probability of going up: ", probability[0][1])
    return trained.predict([inputTuple])


""" Get prediction by random Forest"""

def RandomForestPredict(inputTuple):
    trained = pickle.load(open("SentimentRandomForestPredict5.sav", 'rb'))
    probability = trained.predict_proba([inputTuple])[0][1]
    predict_result = trained.predict([inputTuple])
    if (predict_result == 1):
        print "Probability of going up is: ", probability
        print ("Based on the history and Random Forest Algorithm, this stock will go up at the next day")
    else:
        print ("Probability of going up is: ", probability)
        print ("Based on the history and Random Forest Algorithm, this stock will go down at the next day")


""" Get prediction by logistic regression """

def LogisticRegressionPredict(inputTuple):
    trained = pickle.load(open("SentimentPredict3.sav", 'rb'))
    probability = trained.predict_proba([inputTuple])[0][1]
    predict_result = trained.predict([inputTuple])
    if (predict_result == 1):
        print "Probability of going up is: ", probability
        print ("Based on the history and Logistic Regression Algorithm, this stock will go up at the next day")
    else:
        print "Probability of going up is: ", probability
        print ("Based on the history and Logistic Regression Algorithm, this stock will go down at the next day")


""" Get probability of a prediction """

def Probability(filename, inputTuple):
    trained = pickle.load(open(filename, 'rb'))
    probability = trained.predict_proba([inputTuple])
    return probability


""" Test for accuracy of prediction based on logistic regression algorithm """

def test(since):
    with open("trainingSetWithSentiment2.txt") as set:
        testF = []
        testL = []
        count = 0
        for feature in set:
            count +=1
            if (count >= since):
                FLDict = json.loads(feature)
                FT = (FLDict["count"]+1, FLDict["ratio"], FLDict["simplePredictResult"])
                testF.append(Predict('SentimentPredict3.sav', FT))
                testL.append(FLDict["label"])
        print "accuracy: ", accuracy_score(testF, testL)



""" Test for accuracy of prediction based on random forest algorithm """

def testForest(since, until):
    with open("trainingSetWithSentiment2.txt") as set:
        testF = []
        testL = []
        count = 0
        for feature in set:
            count +=1
            if (count >= since and count < until):
                FLDict = json.loads(feature)
                FT = (FLDict["count"]+1, FLDict["ratio"], FLDict["simplePredictResult"])
                testF.append(Predict('SentimentRandomForestPredict5.sav', FT))
                testL.append(FLDict["label"])
        print "accuracy: ", accuracy_score(testF, testL)


""" Predict based on the latest features in data set by random forest machine"""

def predictLastByRandomForest():
    with open("trainingSetWithSentiment2.txt") as set:
        lines = set.readlines()
        latestData = lines[-1]
        FLDict = json.loads(latestData)
        FT = (FLDict["count"] + 1, FLDict["ratio"], FLDict["simplePredictResult"])
        predict_result = Predict('SentimentRandomForestPredict5.sav', FT)[0]
        probability_of_up = Probability('SentimentRandomForestPredict5.sav', FT)[0][1]
        if (predict_result == 1):
            print "Probability of going up is: ", probability_of_up
            print ("Based on the history and Random Forest Algorithm, this stock will go up at the next day")
        else:
            print ("Probability of going up is: ", probability_of_up)
            print ("Based on the history and Random Forest Algorithm, this stock will go down at the next day")


""" Predict based on the latest features in data set by logistic regression machine """

def predictLastByLogisticRegression():
    with open("trainingSetWithSentiment2.txt") as set:
        lines = set.readlines()
        latestData = lines[-1]
        FLDict = json.loads(latestData)
        FT = (FLDict["count"] + 1, FLDict["ratio"], FLDict["simplePredictResult"])
        predict_result = Predict('SentimentPredict3.sav', FT)[0]
        probability_of_up = Probability('SentimentPredict3.sav', FT)[0][1]
        if (predict_result == 1):
            print "Probability of going up is: ", probability_of_up
            print ("Based on the history and Logistic Regression Algorithm, this stock will go up at the next day")
        else:
            print ("Probability of going up is: ", probability_of_up)
            print ("Based on the history and Logistic Regression Algorithm, this stock will go down at the next day")
