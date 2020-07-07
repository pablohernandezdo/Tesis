import torch.nn as nn
import torch.nn.functional as F


class ClassConv(nn.Module):
    def __init__(self):
        super(ClassConv, self).__init__()

        self.conv1 = nn.Conv1d(1, 10, 2, stride=2)
        self.conv2 = nn.Conv1d(10, 100, 2, stride=2)
        self.conv3 = nn.Conv1d(100, 500, 2, stride=2)
        self.conv4 = nn.Conv1d(500, 1000, 10)
        self.l1 = nn.Linear(1000, 100)
        self.l2 = nn.Linear(100, 10)
        self.l3 = nn.Linear(10, 1)
        self.p1 = nn.AvgPool1d(3)
        self.p2 = nn.AvgPool1d(5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = wave.view(-1, 1, 6000)
        wave = F.relu(self.conv1(wave))
        wave = self.p1(wave)
        wave = F.relu(self.conv2(wave))
        wave = self.p2(wave)
        wave = F.relu(self.conv3(wave))
        wave = self.p2(wave)
        wave = F.relu(self.conv4(wave))
        wave = wave.squeeze()
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ClassConvBN(nn.Module):
    def __init__(self):
        super(ClassConvBN, self).__init__()

        self.conv1 = nn.Conv1d(1, 10, 2, stride=2)
        self.conv2 = nn.Conv1d(10, 100, 2, stride=2)
        self.conv3 = nn.Conv1d(100, 500, 2, stride=2)
        self.conv4 = nn.Conv1d(500, 1000, 10)
        self.l1 = nn.Linear(1000, 100)
        self.l2 = nn.Linear(100, 10)
        self.l3 = nn.Linear(10, 1)
        self.p1 = nn.AvgPool1d(3)
        self.p2 = nn.AvgPool1d(5)
        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d(500)
        self.bn4 = nn.BatchNorm1d(1000)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = wave.view(-1, 1, 6000)
        wave = self.bn1(F.relu(self.conv1(wave)))
        wave = self.p1(wave)
        wave = self.bn2(F.relu(self.conv2(wave)))
        wave = self.p2(wave)
        wave = self.bn3(F.relu(self.conv3(wave)))
        wave = self.p2(wave)
        wave = self.bn4(F.relu(self.conv4(wave)))
        wave = wave.squeeze()
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)
