import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.l1 = nn.Linear(6000, 3000)
        self.l2 = nn.Linear(3000, 1500)
        self.l3 = nn.Linear(1500, 750)
        self.l4 = nn.Linear(750, 300)
        self.l5 = nn.Linear(300, 10)
        self.l6 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.leaky_relu(self.l1(wave), 0.02)
        wave = F.leaky_relu(self.l2(wave), 0.02)
        wave = F.leaky_relu(self.l3(wave), 0.02)
        wave = F.leaky_relu(self.l4(wave), 0.02)
        wave = F.leaky_relu(self.l5(wave), 0.02)
        wave = F.leaky_relu(self.l6(wave), 0.02)
        return self.sigmoid(wave)


class Classifier_XS(nn.Module):
    def __init__(self):
        super(Classifier_XS, self).__init__()

        self.l1 = nn.Linear(6000, 1000)
        self.l2 = nn.Linear(1000, 500)
        self.l3 = nn.Linear(500, 50)
        self.l4 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.leaky_relu(self.l1(wave), 0.02)
        wave = F.leaky_relu(self.l2(wave), 0.02)
        wave = F.leaky_relu(self.l3(wave), 0.02)
        wave = F.leaky_relu(self.l4(wave), 0.02)
        return self.sigmoid(wave)


class Classifier_S(nn.Module):
    def __init__(self):
        super(Classifier_S, self).__init__()

        self.l1 = nn.Linear(6000, 2000)
        self.l2 = nn.Linear(2000, 500)
        self.l3 = nn.Linear(500, 250)
        self.l4 = nn.Linear(250, 50)
        self.l5 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.leaky_relu(self.l1(wave), 0.02)
        wave = F.leaky_relu(self.l2(wave), 0.02)
        wave = F.leaky_relu(self.l3(wave), 0.02)
        wave = F.leaky_relu(self.l4(wave), 0.02)
        wave = F.leaky_relu(self.l5(wave), 0.02)
        return self.sigmoid(wave)


class Classifier_XL(nn.Module):
    def __init__(self):
        super(Classifier_XL, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.l2 = nn.Linear(5000, 4000)
        self.l3 = nn.Linear(4000, 2000)
        self.l4 = nn.Linear(2000, 1000)
        self.l5 = nn.Linear(1000, 200)
        self.l6 = nn.Linear(200, 50)
        self.l7 = nn.Linear(50, 10)
        self.l8 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.leaky_relu(self.l1(wave), 0.02)
        wave = F.leaky_relu(self.l2(wave), 0.02)
        wave = F.leaky_relu(self.l3(wave), 0.02)
        wave = F.leaky_relu(self.l4(wave), 0.02)
        wave = F.leaky_relu(self.l5(wave), 0.02)
        wave = F.leaky_relu(self.l6(wave), 0.02)
        wave = F.leaky_relu(self.l7(wave), 0.02)
        wave = F.leaky_relu(self.l8(wave), 0.02)
        return self.sigmoid(wave)


class Classifier_XXL(nn.Module):
    def __init__(self):
        super(Classifier_XXL, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.bn1 = nn.BatchNorm1d(num_features=5000)
        self.l2 = nn.Linear(5000, 4000)
        self.bn2 = nn.BatchNorm1d(num_features=4000)
        self.l3 = nn.Linear(4000, 3000)
        self.bn3 = nn.BatchNorm1d(num_features=3000)
        self.l4 = nn.Linear(3000, 2000)
        self.bn4 = nn.BatchNorm1d(num_features=2000)
        self.l5 = nn.Linear(2000, 1000)
        self.bn5 = nn.BatchNorm1d(num_features=1000)
        self.l6 = nn.Linear(1000, 500)
        self.bn6 = nn.BatchNorm1d(num_features=500)
        self.l7 = nn.Linear(500, 250)
        self.bn7 = nn.BatchNorm1d(num_features=250)
        self.l8 = nn.Linear(250, 100)
        self.bn8 = nn.BatchNorm1d(num_features=100)
        self.l9 = nn.Linear(100, 50)
        self.bn9 = nn.BatchNorm1d(num_features=50)
        self.l10 = nn.Linear(50, 25)
        self.bn10 = nn.BatchNorm1d(num_features=25)
        self.l11 = nn.Linear(25, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.leaky_relu(self.bn1(self.l1(wave)), 0.02)
        wave = F.leaky_relu(self.bn2(self.l2(wave)), 0.02)
        wave = F.leaky_relu(self.bn3(self.l3(wave)), 0.02)
        wave = F.leaky_relu(self.bn4(self.l4(wave)), 0.02)
        wave = F.leaky_relu(self.bn5(self.l5(wave)), 0.02)
        wave = F.leaky_relu(self.bn6(self.l6(wave)), 0.02)
        wave = F.leaky_relu(self.bn7(self.l7(wave)), 0.02)
        wave = F.leaky_relu(self.bn8(self.l8(wave)), 0.02)
        wave = F.leaky_relu(self.bn9(self.l9(wave)), 0.02)
        wave = F.leaky_relu(self.bn10(self.l10(wave)), 0.02)
        wave = F.leaky_relu(self.l11(wave), 0.02)
        return self.sigmoid(wave)


class Classifier_XXXL(nn.Module):
    def __init__(self):
        super(Classifier_XXXL, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.l2 = nn.Linear(5000, 4000)
        self.l3 = nn.Linear(4000, 3000)
        self.l4 = nn.Linear(3000, 2000)
        self.l5 = nn.Linear(2000, 1000)
        self.l6 = nn.Linear(1000, 500)
        self.l7 = nn.Linear(500, 250)
        self.l8 = nn.Linear(250, 100)
        self.l9 = nn.Linear(100, 50)
        self.l10 = nn.Linear(50, 20)
        self.l11 = nn.Linear(20, 10)
        self.l12 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.leaky_relu(self.l1(wave), 0.02)
        wave = F.leaky_relu(self.l2(wave), 0.02)
        wave = F.leaky_relu(self.l3(wave), 0.02)
        wave = F.leaky_relu(self.l4(wave), 0.02)
        wave = F.leaky_relu(self.l5(wave), 0.02)
        wave = F.leaky_relu(self.l6(wave), 0.02)
        wave = F.leaky_relu(self.l7(wave), 0.02)
        wave = F.leaky_relu(self.l8(wave), 0.02)
        wave = F.leaky_relu(self.l9(wave), 0.02)
        wave = F.leaky_relu(self.l10(wave), 0.02)
        wave = F.leaky_relu(self.l11(wave), 0.02)
        wave = F.leaky_relu(self.l12(wave), 0.02)
        return self.sigmoid(wave)
