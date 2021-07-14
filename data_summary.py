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

start, end, label = fetch_data("/home/rj/AMI/amicorpus/ES2002a/audio/half.lab")
mfcc = extract_mfcc("/home/rj/AMI/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav", start, end,label)
pitch = extract_pitch("/home/rj/AMI/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav", start, end,label)
feat = combine_feats(mfcc, pitch)
rows, columns = feat.size()
# print(rows,columns)
feat = torch.reshape(feat, (rows, 1, columns))
print("Total Number:",len(feat))
y_train, y_test = get_labels(feat, start, end, label)
feat_train, feat_test = split_feats(feat, len(y_train), len(y_test))
print("Train size", feat_train.size()[0], "Test size",feat_test.size()[0])

