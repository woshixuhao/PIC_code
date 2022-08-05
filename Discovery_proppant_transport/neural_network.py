import numpy as np
import torch.nn.functional as F
import torch
import random
import os
from torch.autograd import Variable
from torch.nn import Linear,Tanh,Sequential
import torch.nn as nn
from matplotlib import pyplot as plt


def mish(x):
    M=x*torch.tanh(torch.log(1+torch.exp(x)))
    return M


#自定义损失函数

class ANN(nn.Module):
    def __init__(self,in_neuron,hidden_neuron,out_neuron):
        super(ANN, self).__init__()
        self.layer1 = nn.Linear(in_neuron,hidden_neuron)
        self.layer2 = nn.Linear(hidden_neuron, hidden_neuron)
        self.layer3 = nn.Linear(hidden_neuron, hidden_neuron)
        self.layer4 = nn.Linear(hidden_neuron, hidden_neuron)
        self.layer5 = nn.Linear(hidden_neuron, out_neuron)

    def forward(self, x):
        x=self.layer1(x)
        x=F.softplus(x)
        x=self.layer2(x)
        x=F.softplus(x)
        x=self.layer3(x)
        x=F.softplus(x)
        x=self.layer4(x)
        x=F.softplus(x)
        x=self.layer5(x)

        return x

class ANN_sin(nn.Module):
    def __init__(self,in_neuron,hidden_neuron,out_neuron):
        super(ANN_sin, self).__init__()
        self.layer1 = nn.Linear(in_neuron,hidden_neuron)
        self.layer2 = nn.Linear(hidden_neuron, hidden_neuron)
        self.layer3 = nn.Linear(hidden_neuron, hidden_neuron)
        self.layer4 = nn.Linear(hidden_neuron, hidden_neuron)
        self.layer5 = nn.Linear(hidden_neuron, out_neuron)

    def forward(self, x):
        x=self.layer1(x)
        x=torch.sin(x)
        x=self.layer2(x)
        x=torch.sin(x)
        x=self.layer3(x)
        x=torch.sin(x)
        x=self.layer4(x)
        x=torch.sin(x)
        x = self.layer5(x)
        return x


class PINNLossFunc(nn.Module):
    def __init__(self,h_data_choose):
        super(PINNLossFunc,self).__init__()
        self.h_data=h_data_choose
        return

    def forward(self,prediction):
        f1=torch.pow((prediction-self.h_data),2).sum()
        MSE=f1
        return MSE

class DerivativeLossFunc(nn.Module):
    def __init__(self,h_data_choose,Hx_choose):
        super(DerivativeLossFunc,self).__init__()
        self.h_data=h_data_choose
        self.Hx_choose=Hx_choose
        return

    def forward(self,prediction,Hx):
        f1=torch.pow((prediction-self.h_data),2).sum()+torch.pow((Hx-self.Hx_choose),2).sum()
        MSE=f1
        return MSE



def random_data(total, choose,choose_validate,x,t,un,x_num,t_num,seed_n=525):
    data=torch.zeros(2)
    h_data=torch.zeros([total,1])
    database=torch.zeros([total,2])
    num=0

    for j in range(x_num):
        for i in range(t_num):
            data[0]=x[j]
            data[1]=t[i]
            h_data[num]=un[j,i]
            database[num]=data
            num+=1

    data_array = np.arange(0, total-1, 1)
    np.random.seed(seed_n)
    np.random.shuffle(data_array)
    h_data_choose = torch.zeros([choose, 1])
    database_choose = torch.zeros([choose, 2])
    h_data_validate= torch.zeros([choose_validate, 1])
    database_validate = torch.zeros([choose_validate, 2])
    num = 0
    for i in range(choose):
        index=data_array[i]
        h_data_choose[i] = h_data[index]
        database_choose[i] = database[index]
    for i in range(choose_validate):
        index=data_array[choose + i]
        h_data_validate[i] = h_data[index]
        database_validate[i] = database[index]
    return h_data_choose,h_data_validate,database_choose,database_validate
