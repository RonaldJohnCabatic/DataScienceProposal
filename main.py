import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
import requests
import matplotlib.pyplot as plt
import math
from utils import *
from models import *

start, end, label = fetch_data("audio/half.lab") # Fetch start and end timestamps for segment labels
mfcc = extract_mfcc("audio/ES2002a.Mix-Headset.wav", start, end,label) # Extract MFCC with 13 coefficients using 25ms window with 10 or 15 ms overlap
pitch = extract_pitch("audio/ES2002a.Mix-Headset.wav", start, end,label) # Extract Pitch using 25ms window with 10 or 15 ms overlap
feat = combine_feats(mfcc, pitch) # Combine extracted features
rows, columns = feat.size() # Size check for input resize to neural network
# print(rows,columns)
feat = torch.reshape(feat, (rows, 1, columns)) # Reshape input
print("Total Number:",len(feat))
y_train, y_test = get_labels(feat, start, end, label) # Get corresponding labels according to timestamps
feat_train, feat_test = split_feats(feat, len(y_train), len(y_test)) # Get corresponding features according to timestamps
print("Train size", feat_train.size()[0], "Test size",feat_test.size()[0]) 


for i in range(3):
    if i == 0:
        model = CNN_2L_FC_R() # Convolution Neural Network (with skip and residual layer)
        model_type = "CNN_2L_FC_R"
    elif i == 1:
        model = CNN_2L_Drop_R() # Convolution Neural Network (with skip and residual layer)
        model_type = "CNN_2L_Drop_R"
    elif i == 2:
        model = CNN_2L_DFC_R() # Convolution Neural Network (with skip and residual layer)
        model_type = "CNN_2L_DFC_R"
    # Hyper parameters
    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    epochs = 400

    print("========== Initializing TRAINING:", model_type)
    # For storing epoch values
    loss_rec = []
    train_acc_rec = []
    test_acc_rec = []


    for epoch in range(epochs):
        y_hat = model(feat_train) # Feed input data for label probability prediction
        loss = criterion(y_hat, y_train) # Loss computation

        optimizer.zero_grad() # Reset the optimizer value to 0
        loss.backward() # calculate the loss for Backward pass
        optimizer.step() # Update the weights

        #### Accuracy Calculation
        pred = y_hat.max(dim = 1)[1] # Evaluate the prob to label
        correct = pred.eq(y_train.view_as(pred)).sum().item() # Compare predicted to true label and count correctly predicted label
        train_accuracy = 100*correct/len(y_train) # Train accuracy computation %
        model_acc = accuracy(model, feat_test, y_test) # Test accuracy computation %
        model.train()
        #### Printing of details
        if ((epoch % 50) == 0): 
            print("Epoch: " + str(epoch+1), "LOSS:", loss.item() ,"TRAIN:",train_accuracy,"TEST:",model_acc)

        ##### Save models
        if epoch > 0:
            if train_accuracy > max(train_acc_rec):
                torch.save(model.state_dict(), "models/train_acc-"+str(model_type)+str(epochs)+".pt")
            if model_acc > max(test_acc_rec):
                torch.save(model.state_dict(), "models/test_acc-"+str(model_type)+str(epochs)+".pt")
            if loss.item() < min(loss_rec):
                torch.save(model.state_dict(), "models/less_loss-"+str(model_type)+str(epochs)+".pt")
        ###### Saving details as list
        loss_rec.append(loss.item())
        train_acc_rec.append(train_accuracy)
        test_acc_rec.append(model_acc)


    #### Save details to a file
    np.savetxt("results/train_acc-"+str(model_type)+str(epochs)+".csv", train_acc_rec, delimiter=", ", fmt='%s')
    np.savetxt("results/test_acc-"+str(model_type)+str(epochs)+".csv", test_acc_rec, delimiter=", ", fmt='%s')
    np.savetxt("results/train_loss-"+str(model_type)+str(epochs)+".csv", loss_rec, delimiter=", ", fmt='%s')

    #### For graph
    x = np.arange(0, epoch+1, 1)
    fig, (ax1,ax2) = plt.subplots(2,1)
    # Graph for Accuracy
    ax1.plot(x, train_acc_rec, "x-",label="train")
    ax1.plot(x, test_acc_rec, "o-" ,label="test")
    ax1.set_ylabel("Accuracy %")
    ax1.set_xlabel("Iteration #")
    ax1.legend()
    ax1.set_ylim([0,100])
	# Graph for Loss
    ax2.plot(x, loss_rec, label="loss")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Iteration #")
    plt.savefig("Loss_Accuracy-"+str(model_type)+str(epochs)+".png")
    # plt.show()

