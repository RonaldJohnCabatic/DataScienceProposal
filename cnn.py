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
def fetch_data(filename):
    f = pd.read_csv(filename)
    out = open(filename + "output", "a+")
    start = f["start"].to_numpy()
    end = f["end"].to_numpy()
    label = f["label"].to_numpy()
    return start, end, label



def extract_mfcc(file, start, end, label):
    ### KALDI
    # audio, sr = torchaudio.load(file)
    # print("original", audio.size())
    # end = int(end[len(end)-1])
    # audio = audio[0,:1]
    # print("cropped: ",audio.size())
    # print("total number of samples:", len(audio[0]))
    # print("Sampling rate:", sr)
    # params = {
    # "channel": 0,
    # "dither": 0.0,
    # "window_type": "hanning",
    # "frame_length": 25.0,
    # "frame_shift": 15.0,
    # "remove_dc_offset": False,
    # "round_to_power_of_two": False,
    # "sample_frequency": sr,
    # }
    # mfcc = torchaudio.compliance.kaldi.mfcc(audio)
    ###############
    ###### normal
    audio, sr = sf.read(file, dtype='float32')
    audio = audio.T
    end = int(end[len(end)-1])
    mfcc = librosa.feature.mfcc(y=audio[0:(sr*end)], sr= sr, n_mfcc=13, dtype='float32', hop_length=int(sr*0.015), win_length=int(sr*0.025)).T
    return torch.from_numpy(mfcc)

def get_labels(feat, start, end, y_base):
    label_base = y_base
    label = []
    label_iterator = 0
    for id in range(len(feat)):
        frame_time = id*(0.015)
        if (end[label_iterator] < frame_time):
            label_iterator += 1
        label.append(label_base[label_iterator])
    # labels = torch.from_numpy(np.array(label)).reshape(len(label),1)
    labels = torch.from_numpy(np.array(label)).reshape(len(label))
    
    return labels



start, end, label = fetch_data("/home/rj/AMI/amicorpus/ES2002a/audio/half.lab")
feat = extract_mfcc("/home/rj/AMI/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav", start, end,label)
rows, columns = feat.size()
# print(rows,columns)
feat = torch.reshape(feat, (rows, 1, columns))
print("TRUTH",len(feat))
print("MFCC Extracted:",feat.size(), type(feat))
y = get_labels(feat, start, end, label)
print("Labels Extracted:", y.size(), type(y))


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 1)
        self.pool = nn.MaxPool1d(1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.fc1 = nn.Linear(64*13, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 16)
        self.fc8 = nn.Linear(16, 8)
        self.fc9 = nn.Linear(8, 3)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x + self.pool(x)
        x = F.relu(self.conv2(x))
        x = x + self.pool(x)
        x = x.view(-1, 64*13)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x

learning_rate = 0.001
batch = 32 # 0.505s per iteration
model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
epochs = 10
print("========== Initializing TRAINING:")
for epoch in range(epochs):
    for idx in range(math.ceil(len(feat)/batch)):
        y_hat = model(feat[batch*idx:batch*(idx+1)])
        loss = criterion(y_hat, y[batch*idx:batch*(idx+1)])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # if epoch%10 == 0 or epoch == epochs-1:
    print("Epoch: " + str(epoch+1), "loss:", loss.item())