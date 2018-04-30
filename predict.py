from sklearn import linear_model, decomposition
pca = linear_model.LogisticRegression()
pca.fit([(0,0,0), (1,1,0), (2,2,0), (2,9,0)], [0,0,0,1])
r = pca.predict([(2,20,0)])
probability = pca.predict_proba([(2,5,0)])
print r
print ("probability: ", probability)