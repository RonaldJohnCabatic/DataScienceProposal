import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio

def fetch_data(filename):
    f = pd.read_csv(filename)
    out = open(filename + "output", "a+")
    start = f["start"].to_numpy()
    end = f["end"].to_numpy()
    label = f["label"].to_numpy()
    return start, end, label





def extract_mfcc(file, start, end, label):
    ###################################################
    ### KALDI FEATURE EXTRACTION ## Further study     #
    # audio, sr = torchaudio.load(file)               #
    # print("original", audio.size())                 #
    # end = int(end[len(end)-1])                      #
    # audio = audio[0,:1]                             #
    # print("cropped: ",audio.size())                 #
    # print("total number of samples:", len(audio[0]))#
    # print("Sampling rate:", sr)                     #
    # params = {                                      #
    # "channel": 0,                                   #
    # "dither": 0.0,                                  #
    # "window_type": "hanning",                       #
    # "frame_length": 25.0,                           #      
    # "frame_shift": 15.0,                            #
    # "remove_dc_offset": False,                      #
    # "round_to_power_of_two": False,                 #
    # "sample_frequency": sr,                         #
    # }                                               #
    # mfcc = torchaudio.compliance.kaldi.mfcc(audio)  #
    ###################################################
    ###### normal
    audio, sr = sf.read(file, dtype='float32')
    audio = audio.T
    end = int(end[len(end)-1])
    mfcc = librosa.feature.mfcc(y=audio[0:(sr*end)], sr= sr, n_mfcc=13, dtype='float32', hop_length=int(sr*0.015), win_length=int(sr*0.025)).T
    return torch.from_numpy(mfcc)

def extract_pitch(file, start, end, label):
    audio, sr = sf.read(file, dtype='float32')
    audio = audio.T
    end = int(end[len(end)-1])
    pitch, magnitude = librosa.piptrack(y=audio[0:(sr*end)], sr= sr, hop_length=int(sr*0.015), win_length=int(sr*0.025))
    pitch_all = pitch.T
    magnitude_all = magnitude.T
    it = 0
    f0 = []
    for pitch,magnitude in zip(pitch_all,magnitude_all):
        f0.append(pitch[np.argmax(magnitude)])
    f0 = torch.from_numpy(np.array(f0).reshape(len(f0),1))
    return f0


def get_labels(feat, start, end, y_base):
    label_base = y_base
    label = []
    label_data = {"train": [], "test": []}
    label_iterator = 0
    part = ["train", "test"]
    i = 0
    for id in range(len(feat)):
        frame_time = id*(0.015)
        if (frame_time > 524.39):
            i = 1
        if (end[label_iterator] < frame_time):
            label_iterator += 1
        label.append(label_base[label_iterator])
        label_data[part[i]].append(label_base[label_iterator])
    # labels = torch.from_numpy(np.array(label)).reshape(len(label))
    ##################################################
    ### For checking number of segments per split
    # print("single speaker",label.count(0))
    # print("overlap", label.count(1))
    # print("sil", label.count(2))
    # print("============ DATA SPLIT ============")
    # print("======TRAIN====\nsingle speaker",label_data["train"].count(0))
    # print("overlap", label_data["train"].count(1))
    # print("sil", label_data["train"].count(2))

    # print("======TEST====\nsingle speaker",label_data["test"].count(0))
    # print("overlap", label_data["test"].count(1))
    # print("sil", label_data["test"].count(2))
    # print("============ DATA SPLIT ============")
    ##################################################
    label_train = torch.from_numpy(np.array(label_data["train"])).reshape(len(label_data["train"]))
    label_test = torch.from_numpy(np.array(label_data["test"])).reshape(len(label_data["test"]))
    # return labels
    return label_train, label_test

def combine_feats(mfcc_all,pitch_all):
    print(mfcc_all.size(), pitch_all.size())
    feat = torch.cat((mfcc_all, pitch_all), 1)
    return feat

def split_feats(feat, num_train, num_test):
    feat_train = feat[:num_train]
    feat_test = feat[num_train:]
    return feat_train, feat_test

def accuracy(model, feat, label_t):
    model.eval()
    with torch.no_grad():
        output = model(feat)
        pred = output.max(dim = 1)[1]
        correct = pred.eq(label_t.view_as(pred)).sum().item()
        train_accuracy = 100*correct/len(label_t)
    return train_accuracy

def extract_audio(file, start, end):
    audio, sr = sf.read(file, dtype='float32')
    start=int((start+74)*sr)
    end = int((start+(2*sr)))
    audio = audio[start:end]*10
    sf.write("testfile.wav",audio,sr)



def extract_mfcc_inference(file):
    ###################################################
    ### KALDI FEATURE EXTRACTION ## Further study     #
    # audio, sr = torchaudio.load(file)               #
    # print("original", audio.size())                 #
    # end = int(end[len(end)-1])                      #
    # audio = audio[0,:1]                             #
    # print("cropped: ",audio.size())                 #
    # print("total number of samples:", len(audio[0]))#
    # print("Sampling rate:", sr)                     #
    # params = {                                      #
    # "channel": 0,                                   #
    # "dither": 0.0,                                  #
    # "window_type": "hanning",                       #
    # "frame_length": 25.0,                           #      
    # "frame_shift": 15.0,                            #
    # "remove_dc_offset": False,                      #
    # "round_to_power_of_two": False,                 #
    # "sample_frequency": sr,                         #
    # }                                               #
    # mfcc = torchaudio.compliance.kaldi.mfcc(audio)  #
    ###################################################
    ###### normal
    audio, sr = sf.read(file, dtype='float32')
    audio = audio.T
    mfcc = librosa.feature.mfcc(y=audio, sr= sr, n_mfcc=13, dtype='float32', hop_length=int(sr*0.015), win_length=int(sr*0.025)).T
    return torch.from_numpy(mfcc)


def extract_pitch_inference(file):
    audio, sr = sf.read(file, dtype='float32')
    audio = audio.T
    pitch, magnitude = librosa.piptrack(y=audio, sr= sr, hop_length=int(sr*0.015), win_length=int(sr*0.025))
    pitch_all = pitch.T
    magnitude_all = magnitude.T
    it = 0
    f0 = []
    for pitch,magnitude in zip(pitch_all,magnitude_all):
        f0.append(pitch[np.argmax(magnitude)])
    f0 = torch.from_numpy(np.array(f0).reshape(len(f0),1))
    return f0


def combine_feats_inference(mfcc_all,pitch_all):
    print(mfcc_all.size(), pitch_all.size())
    feat = torch.cat((mfcc_all, pitch_all), 1)
    return feat

def accuracy_inference(model, feat, label_t):
    model.eval()
    with torch.no_grad():
        output = model(feat)
        pred = output.max(dim = 1)[1]
        correct = pred.eq(label_t.view_as(pred)).sum().item()
        train_accuracy = 100*correct/len(label_t)
        print("Data statistics:",correct, "/", len(label_t))
    return pred, correct, train_accuracy