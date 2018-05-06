from sklearn import linear_model, decomposition
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


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

def SimplePredict(filename, changeOverTime):
    trained = pickle.load(open(filename, 'rb'))
    probability = trained.predict_proba([(changeOverTime, 1)])
    return probability[0][1]

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

    filename = 'SentimentPredict2.sav'
    pickle.dump(pca, open(filename, 'wb'))

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

    filename = 'SentimentRandomForestPredict4.sav'
    pickle.dump(pca, open(filename, 'wb'))


def Predict(filename, inputTuple):
    trained = pickle.load(open(filename, 'rb'))
    probability = trained.predict_proba([inputTuple])
    #print ("probability of going up: ", probability[0][1])
    return (trained.predict([inputTuple]))

def RandonForestPredict(inputTuple):
    trained = pickle.load(open("SentimentRandomForestPredict4.sav", 'rb'))
    probability = trained.predict_proba([inputTuple])[0][1]
    predict_result = trained.predict([inputTuple])
    if (predict_result == 1):
        print "Probability of going up is: ", probability
        print ("Based on the history and Random Forest Algorithm, this stock will go up at the next day")
    else:
        print ("Probability of going up is: ", probability)
        print ("Based on the history and Random Forest Algorithm, this stock will go down at the next day")

def LogisticRegressionPredict(inputTuple):
    trained = pickle.load(open("SentimentPredict2.sav", 'rb'))
    probability = trained.predict_proba([inputTuple])[0][1]
    predict_result = trained.predict([inputTuple])
    if (predict_result == 1):
        print "Probability of going up is: ", probability
        print ("Based on the history and Logistic Regression Algorithm, this stock will go up at the next day")
    else:
        print ("Probability of going up is: ", probability)
        print ("Based on the history and Logistic Regression Algorithm, this stock will go down at the next day")

def Probability(filename, inputTuple):
    trained = pickle.load(open(filename, 'rb'))
    probability = trained.predict_proba([inputTuple])
    # print ("probability of going up: ", probability[0][1])
    return probability
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
                testF.append(Predict('SentimentPredict2.sav', FT))
                testL.append(FLDict["label"])
        print "accuracy: ", accuracy_score(testF, testL)


def testForest(since):
    with open("trainingSetWithSentiment2.txt") as set:
        testF = []
        testL = []
        count = 0
        for feature in set:
            count +=1
            if (count >= since):
                FLDict = json.loads(feature)
                FT = (FLDict["count"]+1, FLDict["ratio"], FLDict["simplePredictResult"])
                testF.append(Predict('SentimentRandomForestPredict4.sav', FT))
                testL.append(FLDict["label"])
        print "accuracy: ", accuracy_score(testF, testL)

def predictLastByRandomForest():
    with open("trainingSetWithSentiment2.txt") as set:
        lines = set.readlines()
        latestData = lines[-1]
        FLDict = json.loads(latestData)
        FT = (FLDict["count"] + 1, FLDict["ratio"], FLDict["simplePredictResult"])
        predict_result = Predict('SentimentRandomForestPredict4.sav', FT)[0]
        probability_of_up = Probability('SentimentRandomForestPredict4.sav', FT)[0][1]
        if (predict_result == 1):
            print "Probability of going up is: ", probability_of_up
            print ("Based on the history and Random Forest Algorithm, this stock will go up at the next day")
        else:
            print ("Probability of going up is: ", probability_of_up)
            print ("Based on the history and Random Forest Algorithm, this stock will go down at the next day")

def predictLastByLogisticRegression():
    with open("trainingSetWithSentiment2.txt") as set:
        lines = set.readlines()
        latestData = lines[-1]
        FLDict = json.loads(latestData)
        FT = (FLDict["count"] + 1, FLDict["ratio"], FLDict["simplePredictResult"])
        predict_result = Predict('SentimentPredict2.sav', FT)[0]
        probability_of_up = Probability('SentimentPredict2.sav', FT)[0][1]
        if (predict_result == 1):
            print "Probability of going up is: ", probability_of_up
            print ("Based on the history and Logistic Regression Algorithm, this stock will go up at the next day")
        else:
            print ("Probability of going up is: ", probability_of_up)
            print ("Based on the history and Logistic Regression Algorithm, this stock will go down at the next day")



#SentimentRandomForestTrain(150)
#testForest(400)
#SentimentTrain(200)
#test(200)

#predictLastByRandomForest()
#predictLastByLogisticRegression()