import numpy as np
from matplotlib import pyplot as plt
from torch.distributed.run import parse_args
import config
import torch
import platform
import torch.nn as nn
from mlxtend.data import loadlocal_mnist
from torch.utils import data

def my_relu(X):

    # 将数据存储与NumPy数组并返回
    return nn.ReLU(X)

# 1 初始化模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

# 2 数据库的导入
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


# 重写dataset
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
        # print(x)
        return x, label

    def __len__(self):
        return len(self.label)

# 搭建神经网络
class Classifier(nn.Module):
    # 初始化函数，对网路的输入层、隐含层、输出层的大小和使用的函数进行了规定。
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, num_outputs)

    def forward(self, x):
        x = x.view(-1, num_inputs)
        x = my_relu(self.fc1(x))
        x = self.fc2(x)
        return x



# 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            test_loss += loss.item() * len(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Average Loss: {:.6f} Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy

# 描绘预测函数
def performance(train_loss_history, train_accuracy_history, test_loss_history, test_accuracy_history):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].plot(train_loss_history, label="train_loss")
    ax[0].plot(test_loss_history, label="test_loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")
    ax[0].legend()

    ax[1].plot(train_accuracy_history, label="train_accuracy")
    ax[1].plot(test_accuracy_history, label="test_accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("accuracy")
    ax[1].legend()
    plt.savefig('result.png')

def main():
    args = config.args
    device = torch.device('cpu' if args.cpu else 'cuda')
    print(device)

    # Set directory paths
    DATA_PATH = 'data'
    MODELS_PATH = 'models'
    RESULTS_PATH = 'results'

    # Load dataset using my_dataset
    train_dataset = my_dataset(X_train, y_train)
    test_dataset = my_dataset(X_test, y_test)

    # Create data loaders
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model = Classifier()

    # Move model to device
    model.to(device)

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Train model
    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch)
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_loss, test_accuracy = test(model, device, test_loader)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

    # Save model
    model_params = {
        'state_dict': model.state_dict(),
        'num_hiddens': args.num_hiddens,
        'lr': args.lr,
        'num_epochs': args.epochs
    }
    torch.save(model_params, MODELS_PATH + '/model.pth')

    # Visualize results
    performance(train_loss_history, train_accuracy_history, test_loss_history, test_accuracy_history)

if __name__ == '__main__':
    main()




