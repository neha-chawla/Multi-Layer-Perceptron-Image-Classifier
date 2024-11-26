import numpy as np
import pandas as pd
import csv
import random
import sys

train_images = []
train_labels = []
test_images = []
test_labels = []
training = []
validation = []
batches = []
trainimage = sys.argv[1]
trainlabel = sys.argv[2]
testimage = sys.argv[3]

with open(trainimage) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        train_images.append([int(x) for x in row])

with open(trainlabel) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        train_labels.append(int(row[0]))

with open(testimage) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        test_images.append([int(x) for x in row])
        test_labels.append(1)
        
j = 0

for i in train_images:
    train_images[j] = [x/255 for x in i]
    j += 1

j = 0

for i in test_images:
    test_images[j] = [x/255 for x in i]
    j += 1

train_images = np.array(train_images)
test_images = np.array(test_images)

img_df = {f"z{i}":train_images[:,i] for i in range(784)}
img_df["label"] = train_labels
df_img_train = pd.DataFrame(img_df)

img_df = {f"z{i}":test_images[:,i] for i in range(784)}
img_df["label"] = test_labels
df_img_test = pd.DataFrame(img_df)

X = df_img_train.iloc[:,:-1].values
X_test = df_img_test.iloc[:,:-1].values
Y = df_img_train.iloc[:,-1].values
Y_test = df_img_test.iloc[:,-1].values

temp = []
cutoff = int(X.shape[0] * (1/6) * (-1))
X_train, X_val = X[:cutoff], X[cutoff:]
y_train, y_val = Y[:cutoff], Y[cutoff:]

inputsize = (len(X[0]), 100)
outputsize = (200, 10)

class RegularLayer:
    def __init__(self):
        pass
    
    def forwardpropagation(self, inputtensor):
        return inputtensor
    
    def backwardpropagation(self, inputtensor, go):
        return go @ np.eye(len(inputtensor[0]))

class ReluLayer(RegularLayer):
    def __init__(self):
        pass
    
    def forwardpropagation(self, inputtensor):
        for i in range(len(inputtensor)):
            for j in range(len(inputtensor[i])):
                if inputtensor[i][j] < 0:
                    inputtensor[i][j] = 0
        return inputtensor
    
    def backwardpropagation(self, inputtensor, go):
        for i in range(len(inputtensor)):
            for j in range(len(inputtensor[i])):
                if inputtensor[i][j] <= 0:
                    inputtensor[i][j] = 0
                else:
                    inputtensor[i][j] = 1
        return inputtensor * go

class DenseLayer(RegularLayer):
    def __init__(self, inputtensor, outputtensor, lr=0.1):
        self.lr = lr
        val = 2 / (inputtensor + outputtensor)
        self.wmatrix = np.random.normal(0.0, np.sqrt(val), (inputtensor,outputtensor))
        self.bvector = np.zeros(outputtensor)
        
    def forwardpropagation(self,inputtensor):
        out = inputtensor @ self.wmatrix
        return out + self.bvector
    
    def backwardpropagation(self,inputtensor,go):
        value = self.wmatrix.T
        self.bvector = self.bvector - self.lr * go.mean(axis=0) * len(inputtensor[0])
        self.wmatrix = self.wmatrix - self.lr * (inputtensor.T @ go) 
        return go @ value

layerinfo = {}
layerlist = []
layerlist.append(DenseLayer(inputsize[0],inputsize[1]))
layerlist.append(ReluLayer())
layerlist.append(DenseLayer(inputsize[1],outputsize[0]))
layerlist.append(ReluLayer())
layerlist.append(DenseLayer(outputsize[0],outputsize[1]))

def compute(layerlist, X):
    firstpass = []
    for i in layerlist:
        firstpass.append(i.forwardpropagation(X))
        X = firstpass[len(firstpass)-1]
    return firstpass

for epoch in range(5):
    order = np.arange(len(X))
    random.shuffle(order)
    for i in range(0, len(X) - 31, 32):
        batches.append([X[order[i:i + 32]], Y[order[i:i + 32]]])
    for i in batches:
        x = i[0]
        y = i[1]
        
        firstpass = compute(layerlist, x)

        layer_inputs = [x] + firstpass
        y_pred = firstpass[-1]
        for i in reversed(range(len(layerlist))):
            layerinfo[layerlist[i]] = layer_inputs[i]

        correctlabels = np.zeros(y_pred.shape)
        for i in range(len(correctlabels)):
            correctlabels[i][y[i]] = 1

        numerator = np.exp(y_pred)
        denominator = np.exp(y_pred).sum(axis=-1,keepdims=True)

        lg = ((numerator/denominator) - correctlabels) / len(y_pred[0])
        for i in layerinfo.keys():
            lg = i.backwardpropagation(layerinfo[i],lg)
    traincheck = np.mean(compute(layerlist, X_train)[-1].argmax(axis=-1) == y_train)
    training.append(traincheck)
    valcheck = np.mean(compute(layerlist, X_val)[-1].argmax(axis=-1) == y_val)
    validation.append(valcheck)

final = []
for i in layerlist:
    final.append(i.forwardpropagation(X_test))
    X_test = final[len(final)-1]
output = final[-1].argmax(axis=-1)

with open("test_predictions.csv", 'w') as csv_file:
    pass

with open('test_predictions.csv', 'a') as csv_file:
    for i in output:
        csv_file.write(str(i) + "\n")

