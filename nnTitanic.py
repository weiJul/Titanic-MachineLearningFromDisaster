from preprocessingData import getData
import numpy as np
import random

from netVan import NetVan
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

# seed
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

iterations = 1000
learn = 0.001

# model parameters

outputLayer = 1

# feature_engineering = False #(7 classes)
feature_engineering = True #(9 classes)

nomalizeData = True

# get data
# trainDataNp, trainTargetNp, testDataNp, testTargetNp, inputLayer = getDataAdvanced()
trainDataNp, trainTargetNp, testDataNp, inputLayer = getData(feature_engineering, nomalizeData)


# data to torch
trainValDataTorch = torch.from_numpy(trainDataNp).float()
trainValTargetTorch = torch.from_numpy(trainTargetNp).float()
testDataTorch = torch.from_numpy(testDataNp).float()

# model for prediction
model = NetVan(inputLayer, outputLayer)

criterion = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learn, momentum=0.3) # step_size=1200, learn = 0.1
# optimizer = torch.optim.Adam(model.parameters(), lr=learn) # step_size=1200, learn = 0.1

trainTargetTorch = trainValTargetTorch.unsqueeze(1)
bestAccVal = 0
for i in range(iterations):
    # if i%200==0:
        # random train
    torch.manual_seed(i)
    trainDataTorch = trainValDataTorch[torch.randperm(trainValDataTorch.size()[0])][:700]
    torch.manual_seed(i)
    trainTargetTorch = trainValTargetTorch[torch.randperm(trainValDataTorch.size()[0])][:700]
    # random val
    torch.manual_seed(i)
    valDataTorch = trainValDataTorch[torch.randperm(trainValDataTorch.size()[0])][700:]
    torch.manual_seed(i)
    valTargetTorch = trainValTargetTorch[torch.randperm(trainValDataTorch.size()[0])][700:]

    trainTargetTorch = trainTargetTorch.unsqueeze(1)

    # training
    model.train()
    net_out_train = model(trainDataTorch)
    loss = criterion(net_out_train, trainTargetTorch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # validation
    model.eval()
    net_out_val = model(valDataTorch)
    accTrain = np.round(accuracy_score(np.around(net_out_train.detach().numpy()).astype(int), trainTargetTorch.detach().numpy().astype(int)), 2)
    accVal = np.round(accuracy_score(np.around(net_out_val.detach().numpy()).astype(int), valTargetTorch.detach().numpy().astype(int)), 2)
    if accVal > bestAccVal:
        bestAccVal = accVal
        torch.save(model.state_dict(), "model/titanic.pth")
    if i%100==0:
        print("accTrain: ",accTrain)
        print("accVal: ",accVal)
        print("bestAccVal: ",bestAccVal)
        print()

# final test
model.load_state_dict(torch.load("model/titanic.pth"))
model.eval()
net_out_test = model(testDataTorch)
net_out_test = np.around(net_out_test.detach().numpy()).astype(int)

# write submission file
id = 892
with open("data/nnSubmission.csv", "w") as f:
    f.write("PassengerId,Survived\n")
    for i in net_out_test:
        f.write(str(id)+","+str(i[0])+"\n")
        id+=1


################################################
# results:
################################################
# parameters: SGD(lr=0,01; momentum=0.3); 9 features; data normalisation
# bestAccVal:  0.9
# testAcc: 0.78
################################################
################################################
