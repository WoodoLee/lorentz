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
import boost_util
from matplotlib import pyplot as plt
from matplotlib import cm


#####
#Data preparing
#####

dfInData  = pd.read_pickle("./data/projVec.pkl")
dfLabel = pd.read_pickle("./data/true.pkl")

dfIn = pd.concat([dfInData,dfLabel],axis=1, ignore_index=True)

print(len(dfIn))

train_X, test_X, train_Y, test_Y = train_test_split(dfInData, dfLabel, test_size=0.2)


print(len(train_X))
print(len(test_X))

# 훈련 데이터 텐서 변환
train_X = torch.tensor(train_X.values)
train_Y = torch.tensor(train_Y.values)

# 테스트 데이터 텐서 변환
test_X = torch.tensor(test_X.values)
test_Y = torch.tensor(test_Y.values)


train = TensorDataset(train_X, train_Y)
# print(train[0])

train_loader = DataLoader(train, batch_size=1, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)

    # 인스턴스 생성
model = Net()

# 오차함수 객체
criterion = nn.CrossEntropyLoss()

# 최적화를 담당할 객체
# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 학습 시작
for epoch in range(300):
    total_loss = 0
    # 분할해 둔 데이터를 꺼내옴
    for train_x, train_y in train_loader:
        # 계산 그래프 구성
        train_x, train_y = Variable(train_x), Variable(train_y)
        # 경사 초기화
        optimizer.zero_grad()
        # 순전파 계산
        output = model(train_x)
        # loss = criterion(output, train_y)
        loss = criterion(output, torch.max(train_y, 1)[1])

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
    # 50회 반복마다 누적오차 출력
    if (epoch+1) % 50 == 0:
        print(epoch+1, total_loss)