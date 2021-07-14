import torch
import torch.nn.functional as F
import torch.nn as nn



class CNN_2L_FC_N(nn.Module):
    def __init__(self):
        super(CNN_2L_FC_N, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 1)
        self.pool = nn.MaxPool1d(1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.fc1 = nn.Linear(64*14, 16)
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
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64*14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = F.relu(self.fc3(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x

class CNN_2L_Drop_N(nn.Module):
    def __init__(self):
        super(CNN_2L_Drop_N, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 1)
        self.pool = nn.MaxPool1d(1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.fc1 = nn.Linear(64*14, 16)
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
        x = x + F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64*14)
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

class CNN_2L_DFC_N(nn.Module):
    def __init__(self):
        super(CNN_2L_DFC_N, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 1)
        self.pool = nn.MaxPool1d(1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.fc1 = nn.Linear(64*14, 16)
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
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64*14)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.dropout(F.relu(self.fc4(x)))
        x = F.relu(self.fc5(x))
        x = self.dropout(F.relu(self.fc6(x)))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x

class CNN_2L_FC_R(nn.Module):
    def __init__(self):
        super(CNN_2L_FC_R, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 1)
        self.pool = nn.MaxPool1d(1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.fc1 = nn.Linear(64*14, 16)
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
        x = x + F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64*14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = x
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x+y))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x

class CNN_2L_Drop_R(nn.Module):
    def __init__(self):
        super(CNN_2L_Drop_R, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 1)
        self.pool = nn.MaxPool1d(1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.fc1 = nn.Linear(64*14, 16)
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
        x = x + F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64*14)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        y = x
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x+y)))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x

class CNN_2L_DFC_R(nn.Module):
    def __init__(self):
        super(CNN_2L_DFC_R, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 1)
        self.pool = nn.MaxPool1d(1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.fc1 = nn.Linear(64*14, 16)
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
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64*14)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        y = x
        x = F.relu(self.fc3(x))
        x = self.dropout(F.relu(self.fc4(x)))
        x = F.relu(self.fc5(x))
        x = self.dropout(F.relu(self.fc6(x)))
        x = F.relu(self.fc7(x+y))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x
