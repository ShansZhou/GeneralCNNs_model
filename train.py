import numpy as np
import matplotlib.pyplot as plt
import time


import Data_loader
import General_CNN_models as gcnn



# load dataset
train_loader, test_loader = Data_loader.loadMNIST()
print("dataset loaded")

# initial model
numOfclasses = 10
myLeNet = gcnn.LeNet(numOfclasses)


# train
epoch = 1
batchsize = 4


for i in range(epoch):
    correct = 0
    count =0
    for n, (batch_X, batch_Y) in enumerate(train_loader):
        
        X = np.array(batch_X)
        Y = np.array(batch_Y)
        X_norm = X/255.0
        myLeNet.train(X_norm, Y)
        
        predictions = myLeNet.predict(X_norm)
        
        correct += np.sum(predictions==Y)
        
        count+=1
        
    accuracy = correct/(count*batchsize)
    print("Epoch: %d -> accuracy: %.2f " % (i, accuracy))