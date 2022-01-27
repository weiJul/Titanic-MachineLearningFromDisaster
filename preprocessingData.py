import pandas as pd
from sklearn import preprocessing
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def featureEngineering(data):
    # replce String
    def substrings_in_string(big_string, substrings):
        for cnt, substring in enumerate(substrings):
            if substring in big_string:
                return cnt

    substrings_in_string

    ###############################################################
    # transform column Name => Title
    ###############################################################

    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                  'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
                  'Don', 'Jonkheer']

    data['Title'] = data['Name'].map(lambda x: substrings_in_string(x, title_list))

    # replacing all titles with mr, mrs, miss, master
    def replace_titles(x):
        title = x['Title']
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Countess', 'Mme']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title == 'Dr':
            if x['Sex'] == 'Male':
                return 'Mr'
            else:
                return 'Mrs'
        else:
            return title

    data['Title'] = data.apply(replace_titles, axis=1)

    # print(train.head())

    ###############################################################
    # transform column Cabin => Deck
    ###############################################################

    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', '']

    data['Cabin'] = data['Cabin'].astype(str)

    data['Deck'] = data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

    # print(dfTrain['Deck'])

    ###############################################################
    # transform column Sex => Sex (int)
    ###############################################################

    def setToNum(sex):
        if sex == 'male':
            return 0
        else:
            return 1

    data['Sex'] = data['Sex'].map(lambda x: setToNum(x))



    ###############################################################
    # transform column Embarked => Embarked (int)
    ###############################################################
    embarked_list = set()
    [embarked_list.update([str(x)]) for x in data['Embarked']]


    embarked_list = ['C','Q','S','nan']
    data['Embarked'] = data['Embarked'].map(lambda x: substrings_in_string(str(x), embarked_list))

    ###############################################################
    # remove nan form Age
    ###############################################################
    data['Age'] = [0 if str(x).lower()=='nan' else x for x in data['Age']]

    ###############################################################
    # remove nan form Fare
    ###############################################################
    data['Fare'] = [0 if str(x).lower()=='nan' else x for x in data['Fare']]

    ###############################################################
    # remove Data
    ###############################################################

    data = data.drop(columns=['PassengerId','Cabin', 'Name', 'Ticket'])

    return data



def trainTargetsToNp(data):
    targetsNp = data["Survived"].to_numpy()
    return targetsNp


def inputDataToNp(data):
    # crate array without target
    dataNp = data.to_numpy()
    return dataNp



def getData(feature_engineering, nomalizeData):

    dfTrain = pd.read_csv("./data/train.csv")
    dftest = pd.read_csv("./data/test.csv")

    # modify Data
    dfTrainMod = featureEngineering(dfTrain)
    dfTestMod = featureEngineering(dftest)


    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)

    # create np-array target data
    trainTargetNp = trainTargetsToNp(dfTrainMod)

    # create input data
    dfTrainMod = dfTrainMod.drop(['Survived'], axis=1)

    # remove artificial classes if naive=True
    if not feature_engineering:
        dfTrainMod = dfTrainMod.drop(['Deck', 'Title'], axis=1)
        dfTestMod = dfTestMod.drop(['Deck', 'Title'], axis=1)

    # create np-array input data
    trainDataNp = inputDataToNp(dfTrainMod)
    testDataNp = inputDataToNp(dfTestMod)

    # normalize input data
    if nomalizeData:
        allData = np.append(trainDataNp, testDataNp, axis=0)
        allData = preprocessing.normalize(allData, norm='l2', axis=0)
        trainDataNp = allData[:891]
        testDataNp = allData[891:]

    # get num features for the nn
    inputLayer = len(dfTrainMod.columns)
    print(f"{inputLayer} classes")
    # print(dfTestMod)

    return trainDataNp, trainTargetNp, testDataNp, inputLayer

def getInfo():
    dfTrain = pd.read_csv("./data/train.csv")
    pd.set_option('display.max_columns', None)
    print('dataframe ############################################')
    print(dfTrain)
    print('info ############################################')
    print(dfTrain.info(verbose=True))
    print('class-survived ############################################')
    print(dfTrain[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False))
    print('sex-survived ############################################')
    print(dfTrain[["Survived","Sex"]].groupby(['Sex'], as_index=True).mean().sort_values(by='Survived', ascending=False))
    print('sex-survived ############################################')
    print(dfTrain[["Survived","Parch"]].groupby(['Parch'], as_index=True).mean().sort_values(by='Survived', ascending=False))

    g = sns.FacetGrid(dfTrain, col='Survived')
    g.map(plt.hist, 'Age', bins=30)
    plt.show()

