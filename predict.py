from sklearn import linear_model, decomposition
import json
import pickle

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
    with open("trainingSetWithSentiment.txt") as set:
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

def SentimentPredict(filename, inputTuple):
    trained = pickle.load(open(filename, 'rb'))
    probability = trained.predict_proba([inputTuple])
    print ("probability: ", probability[0][1])
    print (trained.predict([inputTuple]))
    return probability[0][1]


SentimentPredict('SentimentPredict.sav', (47, 5.212097812097813, 0.49358408884307148))