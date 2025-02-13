import matplotlib.pyplot as plt
import torch
import numpy as np
import csv
import pandas as pd
from torch.utils.data import DataLoader,Dataset

import torch.nn as nn
from torch import optim
import time
from sklearn.feature_selection import SelectKBest,chi2


#9、优化：线性相关：计算每一列和标签列的'相关系数'，只留下相关系数最高的k个特征
def get_feature_importance(feature_data, label_data, k =4,column = None):
    model = SelectKBest(chi2, k=k)      #定义一个选择k个最佳特征的函数
    feature_data = np.array(feature_data, dtype=np.float64) #把字符串转换成float型
    X_new = model.fit_transform(feature_data, label_data)   #用这个函数选择k个最佳特征
    #feature_data是特征数据，label_data是标签数据，该函数可以选择出k个特征
    print('x_new', X_new)
    scores = model.scores_                # scores即每一列与结果的相关性
    # 按重要性排序，选出最重要的 k 个
    indices = np.argsort(scores)[::-1]
    if column:                            # 如果需要打印选中的列名字
        k_best_features = [column[i] for i in indices[0:k].tolist()]         # 选中这些列 打印
        print('k best features are: ',k_best_features)
    return X_new, indices[0:k]                  # 返回选中列的特征和他们的下标。

#2、数据预处理
class CovidDataset(Dataset):
    def __init__(self,file_path,mode="train",all_f=False,f_dim=6): #file_path是数据文件路径，mode用来区分训练集验证集还是测试集，all_f=False默认不使用所有列作为特征列，f_dim=6选取相关系数最大的6列作为特征列
        with open(file_path, "r") as f:
            ori_data = list(csv.reader(f))
            column = ori_data[0] #第0行是列名
            # 处理数据
            csv_data = np.array(ori_data[1:])[: , 1: ].astype(float)
            feature = np.array(ori_data[1:])[: , 1:-1] #特征列：np.array(ori_data[1:])去掉第一行，[: , 1:-1]取第二列到倒数第二列
            label_data = np.array(ori_data[1:])[: ,-1] #标签列：np.array(ori_data[1:])去掉第一行，[: , -1]取最后一列
            # 9、优化：根据相关度选取加入训练的特征
            if all_f:
                col = np.array([i for i in range(0,93)])
            else:
                _, col = get_feature_importance(feature, label_data,f_dim,column) #feature是特征列, label_data是标签列,f_dim选取的列数,column是特征名字
            col = col.tolist() #把col转换成列表

        if mode=="train":
            indices = [i for i in range(len(csv_data)) if i%5 != 0]
            data = torch.tensor(csv_data[indices, :-1]) #训练数据和验证数据不包含最后一列？
            self.y=torch.tensor(csv_data[indices,-1]) #提取y，行对应indices，列是最后一列，转成张量！
        elif mode=="val":
            indices = [i for i in range(len(csv_data)) if i % 5 == 0]
            data = torch.tensor(csv_data[indices, :-1])
            self.y = torch.tensor(csv_data[indices, -1])  # 提取y，行对应indices，列是最后一列，转成张量！
        else:
            indices = [i for i in range(len(csv_data))]
            data = torch.tensor(csv_data[indices, :]) #测试数据要最后一列

        data = data[:, col]
        self.data = (data-data.mean(dim=0, keepdim=True))/data.std(dim=0, keepdim=True)
        self.mode = mode

    #取一行数据
    def __getitem__(self, idx):
        if self.mode != "test":
            return self.data[idx].float(), self.y[idx].float()
        else:
            return self.data[idx].float()

    def __len__(self):
        return len(self.data)

#5、模型
class MyModel(nn.Module):
    def __init__(self,inDim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(inDim,64) #第一个隐层的输入维度和输出维度
        self.relu1 = nn.ReLU() #第一个隐层的激活函数
        self.fc2 = nn.Linear(64,1) #输出层的输入维度和输出维度
    #前向传播
    def forward(self,x):
        x=self.fc1(x)
        x=self.relu1(x)
        x=self.fc2(x)

        #解决x的输出的y不是同型的问题
        if len(x.size()) > 1:
            return x.squeeze(1) #去掉第二列

        return x

#7、训练模型（训练+验证）参数：模型，训练集，验证集，设备，轮次，优化器，损失函数，模型保存路径
def train_val(model,train_loader,val_loader,device,epochs,optimizer,loss,save_path):
    model = model.to(device)

    #记录loss
    plt_train_loss = [] #保存所有轮次的loss
    plt_val_loss = [] #plt提醒待会要画图
    min_val_loss =9999999999 #记录最优模型的loss

    #开始训练
    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        start_t = time.time()

        model.train() #进入训练模式
        for batch_x,batch_y in train_loader: #在训练集中取一批数据
            x,target = batch_x.to(device),batch_y.to(device) #放在GPU上面训练
            pred = model(x) #得到预测值
            train_bat_loss = loss(pred, target,model) # 9、优化：损失函数正则化
            train_bat_loss.backward() #梯度回传
            optimizer.step() #更新模型
            optimizer.zero_grad() #重置loss值
            train_loss += train_bat_loss.cpu().item() #把train_bat_loss放在cpu上面（才能和train_loss相加）取数值

        plt_train_loss.append(train_loss / train_loader.__len__()) #将本轮计算的loss加到已有的plt_train_loss上面并除以总轮数

        model.eval() #进入验证模式
        with torch.no_grad(): #不记录梯度
            for batch_x, batch_y in val_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                val_bat_loss = loss(pred, target,model)  # 9、优化：损失函数正则化
                val_loss += val_bat_loss.cpu().item()  # 把train_bat_loss放在cpu上面（才能和train_loss相加）取数值

        plt_val_loss.append(val_loss / val_loader.__len__())  # 将本轮计算的loss加到已有的plt_train_loss上面并除以总轮数

        # 保存模型
        if val_loss < min_val_loss:
            torch.save(model,save_path)
            min_val_loss = val_loss

        # 打印这一轮的训练结果
        print("[%03d/%03d]  %2.2f sec(s) Trainloss:%.6f | Valloss:%.6f"% \
              (epoch,epochs,time.time() - start_t,plt_train_loss[-1],plt_val_loss[-1]))

        #画一下loss图
        plt.plot(plt_train_loss)
        plt.plot(plt_val_loss)
        plt.title("loss图")
        plt.legend(["train","val"])
        plt.show()

#8、测试模型
def evaluate(save_path,test_loader,device,rel_path):
    model = torch.load(save_path).to(device) #把模型加载到设备上
    rel =[] #保存预测结果
    with torch.no_grad(): #不记录梯度
        for x in test_loader: #在测试模式只有x一个参数
            pred = model(x.to(device)) #计算预测值
            rel.append(pred.cpu().item())
    print(rel)

    #保存到文件
    with open(rel_path,"w",newline='') as f: #newline=''解决结果出现重复换行
        csvWriter=csv.writer(f) #写指针
        csvWriter.writerow(["id","tested_positive"])
        for i,value in enumerate(rel): #同时取下标和下标对应的值
            csvWriter.writerow([str(i),str(value)])
    print("文件已经保存到{}".format(rel_path))

#9、优化：根据相关度选取加入训练的特征
all_f = False
if all_f:
    f_dim = 93
else:
    f_dim = 6

#1、
train_f ="covid.train.csv"
test_f ="covid.test.csv"

#3、
train_data = CovidDataset(train_f,"train",all_f=all_f,f_dim=f_dim)
val_data = CovidDataset(train_f,"val",all_f=all_f,f_dim=f_dim)
test_data = CovidDataset(test_f,"test",all_f=all_f,f_dim=f_dim)

#4、取一批数据
batch_size = 16
train_loader = DataLoader(train_data,batch_size = batch_size,shuffle = True) #
val_loader = DataLoader(val_data,batch_size = batch_size,shuffle = True) #
test_loader = DataLoader(test_data,batch_size = 1,shuffle = False) # 测试只有一轮，而且不能打乱！

#6、超参数
device = "cuda" if torch.cuda.is_available() else "cpu" #设备
# print(device)
# 把超参数放进一个字典里面
config = {
    "lr":0.001,
    "epochs":20,
    "momentum":0.9,
    "save_path":"model_save/bst_model.pth",
    "rel_path":"pred.csv"
}

#9、优化：损失函数正则化
def mseloss_with_reg(pred,target,model):
    loss = nn.MSELoss(reduction='mean')
    #计算损失
    regularization_loss=0 #正则loss为0
    for param in model.parameters(): #对模型中的每一个参数
        #使用L2正则项
        regularization_loss+=torch.sum(param ** 2) #计算所有参数的平方
    return loss(pred,target) + 0.00075 *regularization_loss #返回损失函数，在正则项前面乘一个非常小的数，减小regularization_loss太大时带来的影响

model = MyModel(inDim=f_dim).to(device) #把模型放在GPU上
# loss = nn.MSELoss()
loss = mseloss_with_reg
optimizer = optim.SGD(model.parameters(), lr = config["lr"], momentum=config["momentum"])

#7、训练模型（训练+验证）
train_val(model,train_loader,val_loader,device,config["epochs"],optimizer,loss,config["save_path"])

#8、测试模型
# evaluate(config["save_path"],test_loader,device,config["rel_path"])
