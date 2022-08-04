import numpy as np
import torch
import scipy.io as scio
from torch.nn import Linear,Tanh,Sequential
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import os
import torch.nn.functional as func

GPUID="1"
#设置保留小数位数
torch.set_printoptions(precision=7, threshold=None, edgeitems=None, linewidth=None, profile=None)

#读取数据
data_path=r'/home/lthpc/PycharmProjects/wave-equation-DLAG/wave.mat'
data=scio.loadmat(data_path)
un=data.get("u")
x=np.squeeze(data.get("x"))
t=np.squeeze(data.get("t").reshape(1,321))


#x和T
total=51681
choose=500
choose_validate=4000
un_raw=torch.from_numpy(un.astype(np.float32))
data=torch.zeros(2)
h_data=torch.zeros([total,1])
database=torch.zeros([total,2])
num=0

for j in range(161):
    for i in range(321):
        data[0]=x[j]
        data[1]=t[i]
        h_data[num]=un_raw[j,i]#*(1+0.01*np.random.uniform(-1,1))
        database[num]=data
        num+=1


#a =np.load("a-10000.npy")
a = random.sample(range(0, total-1), choose)
np.save("a-500.npy",a)
# temp=[]
# for i in range(total):
#     if i not in a:
#         temp.append(i)
# b=random.sample(temp, choose_validate)
# np.save("b-4000.npy",b)
b=np.load("b-4000.npy")
h_data_choose = torch.zeros([choose, 1])
database_choose = torch.zeros([choose, 2])
h_data_validate= torch.zeros([choose_validate, 1])
database_validate = torch.zeros([choose_validate, 2])
num = 0
for i in a:
    h_data_choose[num] = h_data[i]
    database_choose[num] = database[i]
    num += 1
num=0
for i in b:
    h_data_validate[num] = h_data[i]
    database_validate[num] = database[i]
    num += 1



loss_min=10000
iter_num=3000000
tol=0.015
num=0
alpha_compair = 10000
torch.manual_seed(525)
class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        x = torch.sin(x)
        return x

#定义一个构建神经网络的类
Net=Sequential(
    Linear(2,50),
    Sin(),
    Linear(50,50),
    Sin(),
    Linear(50,50),
    Sin(),
    Linear(50, 50),
    Sin(),
    Linear(50, 1),
)


#自定义损失函数
class PINNLossFunc(nn.Module):
    def __init__(self,h_data):
        super(PINNLossFunc,self).__init__()
        self.h_data=h_data_choose
        return

    def forward(self,prediction):
        f1=torch.pow((prediction-self.h_data),2).sum()
        MSE=f1
        return MSE


os.environ["CUDA_VISIBLE_DEVICES"] = GPUID
netMulti = nn.DataParallel(Net).cuda()
device_ids=range(torch.cuda.device_count())
database_choose = Variable(database_choose.cuda(),requires_grad=True)
database_validate = Variable(database_validate.cuda(),requires_grad=True)
h_data_choose=Variable(h_data_choose.cuda())

# 定义神经网络
#optimizer=torch.optim.LBFGS(Net.parameters(),max_iter=10000,lr=0.02)# 传入网络参数和学习率
optimizer=torch.optim.Adam([
    {'params': Net.parameters()},
    #{'params': theta},
])


for t in range(iter_num):
    optimizer.zero_grad()
    prediction = Net(database_choose)
    prediction_validate = Net(database_validate).cpu().data.numpy()
    a = PINNLossFunc(h_data_choose)
    loss = a(prediction.cuda()) / choose
    loss_validate = np.sum((h_data_validate.data.numpy() - prediction_validate) ** 2) / choose_validate
    loss.backward()
    optimizer.step()

    if t % 1000 == 0:
        print("iter_num: %d      loss: %.8f    loss_validate: %.8f" % (t, loss, loss_validate))
        if int(t / 100) == 300:
            # sign=stop(loss_validate_record)
            # if sign==0:
            #     break
            break
        # if t>2000:
        #     torch.save(Net.state_dict(), "Wave_equation-2000-%d.pkl"%(t))
torch.save(Net.state_dict(), "wave_equation-500.pkl")