import platform

import numpy as np
from matplotlib import pyplot as plt

import config
import torch
import torchvision
from mlxtend.data import loadlocal_mnist
from torch import nn, optim
from torch.utils import data
from d2l import torch as d2l
from IPython import display
from typing import List, Tuple

def my_relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
# 1 读取数据
# 1.1 重写dataset
class my_dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None, target_transform=None):
        # pytorh的训练集必须是tensor形式，可以直接在dataset类中转换，省去了定义transform
        # 转换Y数据类型为长整型
        self.point = torch.from_numpy(x).type(torch.FloatTensor)
        self.label = torch.from_numpy(y).type(torch.FloatTensor)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = self.point[index]
        label = self.label[index]
        return x, label

    def __len__(self):
        return len(self.label)

# 1.2 读取MNIST数据集及对应的标签
if not platform.system() == 'Windows':
    X_train, y_train = loadlocal_mnist(
            images_path='train-images-idx3-ubyte',
            labels_path='train-labels-idx1-ubyte')
    X_test, y_test = loadlocal_mnist(
            images_path='t10k-images-idx3-ubyte',
            labels_path='t10k-labels-idx1-ubyte')

else:
    X_train, y_train = loadlocal_mnist(
        images_path='train-images-idx3-ubyte',
        labels_path='train-labels-idx1-ubyte')
    X_test, y_test = loadlocal_mnist(
        images_path='t10k-images-idx3-ubyte',
        labels_path='t10k-labels-idx1-ubyte')

# 1.2.1 转换为tenser
args=config.args
device=torch.device('cpu' if args.cpu else 'cuda')
mnist_train = my_dataset(X_train, y_train)
mnist_test = my_dataset(X_test, y_test)
train_loader = data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)
test_loader = data.DataLoader(mnist_test, batch_size=args.batch_size, shuffle=False)

# 查看数据集读取情况
# print('train_len: ', len(mnist_train))
# print('test_len: ', len(mnist_test))
# print('label: ', mnist_test[2][1])

# 2 初始化模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]

# 激活函数
class ReLU:
    def __init__(self):
        self.mask = None
        self.out = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        self.out = out
        return out

    def backward(self):
        dx = self.out
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self):
        dx = (1.0 - self.out) * self.out
        return dx


class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))
        self.out = out
        return out

    def backward(self):
        dx = 1 - np.power(self.out, 2)
        return dx


# 定义单隐层全连接神经网络
class Classifier(nn.Module):
    def __init__(self, hidden_layersize=256):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(784, hidden_layersize)
        self.fc2 = nn.Linear(hidden_layersize, 10)

    def forward(self, input):
        x = nn.ReLU()(self.fc1(input))
        x = self.fc2(x)
        return x

# 对Classifier类进行实例化，创建model对象作为我们的神经网络
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier().to(device)

# 前向计算2号
'''
# 前向计算
def forward(X: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor):
    Z1 = torch.matmul(W1, X) + b1
    A1 = my_relu(Z1)
    Z2 = torch.matmul(W2, A1) + b2
    A2 = my_relu(Z2)
    return [Z1, A1, Z2, A2]
  # 前向传播
    def forward(self, x):
        z1 = np.dot(x, self.weight_H) + self.b_H
        a1 = self.A_F1.forward(z1)
        z2 = np.dot(a1, self.weigth_O) + self.b_O
        z = self.A_F2.forward(z2)
        return z1, a1, z2, z
'''


# BP训练



# 准确率函数
def accuracy(pre: torch.Tensor, true: torch.Tensor):
    sum = 0.0
    for y_hat, y in zip(pre, true):
        sum += (y_hat == y)

    return sum/(len(true))


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 50
print(epochs)
train_losses = []
print('开始训练')
for e in range(epochs):
    running_loss = 0
    for points, labels in train_loader:
        #  将数据展开成大小为 [batch_size, 784] 的二维 Tensor
        points = points.view(points.shape[0], -1)
        labels = labels.type(torch.LongTensor)

        points = points.to(device)
        labels = labels.to(device)
        # 前向传播
        out = model(points)
        # 计算代价函数
        loss = criterion(out, labels)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if (e+1) % 10 == 0:
        print("epoch" + str(e+1))
    train_losses.append(running_loss / len(train_loader))
# 验证画出验证的分类效果
else:
    # 预测模式
    model.eval()
    correct = 0
    total = 0.0
    for points, labels in test_loader:
        points = points.view(points.shape[0], -1)
        labels = labels.type(torch.LongTensor)
        points = points.to(device)
        labels = labels.to(device)

        out = model(points)

        prediction = torch.round(out)
        pred_y = prediction.cpu().data.numpy().squeeze()
        target_y = labels.cpu().data.numpy()
        if(pred_y == target_y):
            correct += 1
        total += points.size(0)
    accuracy = correct / total
    print('Accuracy of the network on the 1000 test images: %d %%' % (100 * accuracy))
        # print(pred_y)
        color = []
        for i in range(pred_y.size):
            if pred_y[i] == 0:
                color.append('r')
            else:
                color.append('b')

        target_y = labels.data.numpy()
        plt.title("Output of classification")
        plt.scatter(points.data.numpy()[:, 0], points.data.numpy()[
                    :, 1], c=color, s=40, lw=0, cmap='RdYlGn')
        accuracy = sum((pred_y == target_y).any())/len(test_loader)  # 预测中有多少和真实值一样
        plt.text(1.7, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.show()
        # 将训练误差和测试误差存在两个列表里，后面绘制误差变化折线图用
    # 恢复训练模式
    model.train()
# 绘制loss下降图（针对不同的学习速率请更改lr之后重新创建model对象，以免下一次该代码片段运行的初始化权重为上一次的结果）
plt.title("Train_loss")
plt.plot(train_losses, label='Training loss',)
plt.legend()
plt.show()



# 训练模型
num_epochs, lr = args.num_epochs, args.lr

animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.0, 1.0],
                        legend=['train loss', 'train acc', 'test acc'])

'''
def evaluate_accuracy(net, data_iter):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        with torch.no_grad():
            acc_sum += (net(X).argmax(axis=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train(net, train_iter, loss, num_epochs, lr):
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            train_loss_sum += l.item()
            train_acc_sum += (y_hat.argmax(axis=1) == y).float().sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, (train_loss_sum / n, train_acc_sum / n, test_acc))


def train(model, dataloader, optimizer, criterion: nn.CrossEntropyLoss, lr, num_epochs):


    for epoch in range(num_epochs):
        for i, (X, y) in enumerate(dataloader):
            # 将图片转换为向量
            X = X.reshape((-1, num_inputs)).transpose(0, 1)
            y = y.reshape((-1,)).long()

            # 前向计算
            Z1, A1, Z2, y_hat = model.forwad(X)

            # 计算损失
            batch_loss = criterion(y_hat, y)

            delta3 = z - y
            dw2 = np.dot(a1.T, delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = np.dot(delta3, self.weight_O.T) * self.AF.backward()
            dw1 = np.dot(x.T, delta2)
            db1 = np.sum(delta2, axis=0)
            # dw2 += self.reg_lambda * self.weight_O
            # dw1 += self.reg_lambda * self.weight_H
            model.weight_H -= self.learning_rate * dw1
            self.b_H -= self.learning_rate * db1
            self.weight_O -= self.learning_rate * dw2
            self.b_O -= self.learning_rate * db2

            # 反向传播及更新参数
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # 计算在测试集上的准确率并输出
        test_acc = evaluate_accuracy(model, test_iter)
        print(f"epoch {epoch+1}, test acc {test_acc:.4f}")
'''
# train(net, train_iter, test_iter, loss, num_epochs, lr, animator)

# 模型评估
# test_acc = evaluate_accuracy(net, test_iter)
# print(f'test_acc: {test_acc}')