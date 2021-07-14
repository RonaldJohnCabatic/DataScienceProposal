import pandas as pd
import librosa, os
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


start = float(524.39)
end = float(655.48)
extract_audio("audio/ES2002a.Mix-Headset.wav",start,end)

mfcc = extract_mfcc_inference("testfile.wav") # Extract MFCC with 13 coefficients using 25ms window with 10 or 15 ms overlap
pitch = extract_pitch_inference("testfile.wav") # Extract Pitch using 25ms window with 10 or 15 ms overlap
feat = combine_feats_inference(mfcc, pitch) # Combine extracted features
rows, columns = feat.size() # Size check for input resize to neural network

feat = torch.reshape(feat, (rows, 1, columns)) # Reshape input
label = torch.ones(len(feat), dtype=torch.int)





### Accuracy Calculation
# model = CNN_2L_FC_R() # Convolution Neural Network (with skip and residual layer)
# model = CNN_2L_Drop_R() # Convolution Neural Network (with skip and residual layer)
model = CNN_2L_DFC_R() # Convolution Neural Network (with skip and residual layer)
model_dir = "models_submitted"
model_list = os.listdir(model_dir)
sorted(model_list)
for item in model_list:

    path="/".join([model_dir,item])
    model.load_state_dict(torch.load(path))
    pred, correct, accuracy = accuracy_inference(model, feat, label) # Test accuracy computation %
    print(item.split('.')[0],":",accuracy)
