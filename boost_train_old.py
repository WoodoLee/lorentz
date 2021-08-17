import pretty_errors
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import boost_util as boost
from matplotlib import pyplot as plt
from matplotlib import cm
import argparse


parser = argparse.ArgumentParser(description='Lorentz Boosting Example')
parser.add_argument('--batch-size', type=int, default=513, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

class NetMain(nn.Module):
    def __init__(self):
        super(NetMain, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        # self.fc3 = nn.Linear(513, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        print(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim = 1)

class NetBoost(nn.Module):
    def __init__(self):
        super(NetBoost, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 7)
     
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)



    # 인스턴스 생성

train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.test_batch_size}

if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                    # 'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)


####################
# Data preparing
####################

dfInData  = pd.read_pickle("./data/projVec.pkl")
dfLabel = pd.read_pickle("./data/true.pkl")


# print(dfIn)
train_X, test_X, train_Y, test_Y = train_test_split(dfInData, dfLabel, test_size=0.2)

def catTime(dataIn, dt):
    npTime = [dt]*(len(dataIn))
    dfTime =  pd.Series(npTime)
    return dfTime
dtRest = 0.

train_dfT = catTime(train_X, dtRest)
# print(len(train_X))
# print(len(test_X))

# 훈련 데이터 텐서 변환
train_X = torch.tensor(train_X.values)
train_Y = torch.tensor(train_Y.values)
train_T = torch.tensor(train_dfT.values)

# 테스트 데이터 텐서 변환
test_X = torch.tensor(test_X.values)
test_Y = torch.tensor(test_Y.values)


train = TensorDataset(train_X, train_Y)
# print(train[0])

train_loader = DataLoader(train, **train_kwargs)

# print(mB)

####################
# Training
####################

modelMain = NetMain().to(device)
modelBoost = NetBoost().to(device)
# 오차함수 객체
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# 최적화를 담당할 객체
# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.SGD(modelMain.parameters(), lr=0.001, momentum=0.9)
# optimizer2 = optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)
# optimizer = optimizer.to(device)
print(modelMain)
print(modelBoost)
# 학습 시작
print(torch.cuda.get_device_name(0))

for epoch in range(100):
    total_loss = 0
    # 분할해 둔 데이터를 꺼내옴
    for train_x, train_y in train_loader:
        # 계산 그래프 구성
        # train_x, train_y = Variable(train_x), Variable(train_y)
        # print(train_x.shape)
        train_x = torch.transpose(train_x, 0, 1)
        # print(train_x.shape)
        # print(mB.shape)
        train_x = train_x.float().to(device)
        train_y = train_y.float().to(device)
        train_t = train_T.float().to(device)

        print(train_x.shape)
        print(train_t.shape)
        print(train_t.shape)
        print(train_t)
        
        train_tx = torch.cat([train_x,train_t])
        
        output = model(train_tx)
        loss = criterion(output, torch.max(train_y, 1)[1])
        # print(loss)
        # 오차계산
        # loss = criterion(output, train_y)
        # loss = criterion(output, train_y)
        # loss = criterion(output, torch.Tensor([7]).long()) #
        # 역전파 계산
        loss.backward()
        # 가중치 업데이트
        optimizer.step()
        # 누적 오차 계산
        total_loss += loss.data
    if (epoch+1) % 10 == 0:
        print(epoch+1, total_loss)
        # print('=================== output =========================')
        # print(output, len(output))
        # print('=================== train X =========================')
        # print(train_x, len(train_x[0]))

