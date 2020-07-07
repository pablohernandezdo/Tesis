import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.l1 = nn.Linear(30, 5)
        self.l2 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.leaky_relu(self.l1(wave), 0.02)
        wave = self.l2(wave)
        return self.sigmoid(wave)
