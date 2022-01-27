from preprocessingData import getData, getInfo
from sklearn.tree import DecisionTreeClassifier

getInfo()

# feature_engineering = False #(7 classes)
feature_engineering = True #(9 classes)
nomalizeData = True

trainDataNp, trainTargetNp, testDataNp, _ = getData(feature_engineering, nomalizeData)

# dt
clf = DecisionTreeClassifier(random_state=1234)
model = clf.fit(trainDataNp, trainTargetNp)

# predict the response for test dataset
testDataPred = clf.predict(testDataNp)

for i in testDataPred:
    print(i)

id = 892
with open("data/dtSubmission.csv", "w") as f:
    f.write("PassengerId,Survived\n")
    for i in testDataPred:
        f.write(str(id)+","+str(i)+"\n")
        id+=1

dtAcc = round(clf.score(trainDataNp, trainTargetNp), 2)
print(dtAcc)

################################################
# results:
################################################
# parameters: feature_engineering; nomalizeData
# train acc: 0.99
# test acc: 0.73
################################################
################################################

