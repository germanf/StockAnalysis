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
    print ("probability: ", probability[0][1])
    return probability[0][1]

def SentimentTrain():
    features = []
    labels = []
    with open("trainingSetWithSentiment2.txt") as set:
        for feature in set:
            FLDict = json.loads(feature)
            FT = (FLDict["count"], FLDict["ratio"], FLDict["simplePredictResult"])
            LT = FLDict["label"]
            features.append(FT)
            labels.append(LT)
    pca = linear_model.LogisticRegression()
    pca.fit(features, labels)

    filename = 'SentimentPredict.sav'
    pickle.dump(pca, open(filename, 'wb'))

def SentimentRandomForestTrain():
    features = []
    labels = []
    with open("trainingSetWithSentiment2.txt") as set:
        for feature in set:
            FLDict = json.loads(feature)
            FT = (FLDict["count"], FLDict["ratio"], FLDict["simplePredictResult"])
            LT = FLDict["label"]
            features.append(FT)
            labels.append(LT)
    pca = RandomForestClassifier()
    pca.fit(features, labels)

    filename = 'SentimentRandomForestPredict3.sav'
    pickle.dump(pca, open(filename, 'wb'))


def Predict(filename, inputTuple):
    trained = pickle.load(open(filename, 'rb'))
    probability = trained.predict_proba([inputTuple])
    print ("probability: ", probability[0][1])
    return (trained.predict([inputTuple]))


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
                testF.append(Predict('SentimentPredict.sav', FT))
                testL.append(FLDict["label"])
        print accuracy_score(testF, testL)


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
                testF.append(Predict('SentimentRandomForestPredict3.sav', FT))
                testL.append(FLDict["label"])
        print accuracy_score(testF, testL)

def predictLastByForest():
    with open("trainingSetWithSentiment2.txt") as set:
        lines = set.readlines()
        latestData = lines[-1]
        FLDict = json.loads(latestData)
        FT = (FLDict["count"] + 1, FLDict["ratio"], FLDict["simplePredictResult"])
        predict_result = Predict('SentimentRandomForestPredict3.sav', FT)[0]
        if (predict_result == 1):
            print ("Based on the history, this stock will go up at the next day")
        else:
            print ("Based on the history, this stock will go down at the next day")

def predictLastByLogisitcRegression():
    with open("trainingSetWithSentiment2.txt") as set:
        lines = set.readlines()
        latestData = lines[-1]
        FLDict = json.loads(latestData)
        FT = (FLDict["count"] + 1, FLDict["ratio"], FLDict["simplePredictResult"])
        predict_result = Predict('SentimentRandomForestPredict3.sav', FT)[0]
        if (predict_result == 1):
            print ("Based on the history, this stock will go up at the next day")
        else:
            print ("Based on the history, this stock will go down at the next day")



#SentimentRandomForestTrain()
#testForest(150)
#SentimentTrain()
#test(200)

predictLastByForest()