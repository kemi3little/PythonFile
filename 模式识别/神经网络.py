import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import os
import gzip


def download(filename, url, force=False):
    """Download a file if it is not already downloaded."""
    if force or not os.path.exists(filename):
        print('Downloading %s...' % filename)
        urllib.request.urlretrieve(url, filename)


def load_mnist_images(filename):
    """Load images from MNIST data."""
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28 * 28)
    return data


def load_mnist_labels(filename):
    """Load labels from MNIST data."""
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def preprocess_data(x, y, num_classes):
    x = x.astype(np.float32)
    x /= 255
    y = y.astype(np.int32)
    y_onehot = np.zeros((y.shape[0], num_classes))
    y_onehot[np.arange(y.shape[0]), y] = 1
    return x, y_onehot


class NeuralNetwork():
    def __init__(self, num_input, num_hidden, num_output):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        self.weights1 = np.random.randn(self.num_input, self.num_hidden)
        self.biases1 = np.zeros((self.num_hidden,))
        self.weights2 = np.random.randn(self.num_hidden, self.num_output)
        self.biases2 = np.zeros((self.num_output,))

    def forward(self, x):
        z1 = np.dot(x, self.weights1) + self.biases1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.weights2) + self.biases2
        a2 = self.softmax(z2)
        return a2

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def backward(self, x, y_true, y_pred, learning_rate):
        delta = y_pred - y_true
        dW2 = np.dot(self.a1.T, delta)
        db2 = np.sum(delta, axis=0)
        delta = np.dot(delta, self.weights2.T) * (1 - np.square(self.a1))
        dW1 = np.dot(x.T, delta)
        db1 = np.sum(delta, axis=0)

        self.weights2 -= learning_rate * dW2
        self.biases2 -= learning_rate * db2
        self.weights1 -= learning_rate * dW1
        self.biases1 -= learning_rate * db1

    def train(self, x_train, y_train, epochs, batch_size, learning_rate):
        num_samples = x_train.shape[0]
        for i in range(epochs):
            perm = np.random.permutation(num_samples)
            for j in range(0, num_samples, batch_size):
                indices = perm[j:j + batch_size]
                x_batch = x_train[indices]
                y_batch = y_train[indices]
                self.a1 = np.tanh(np.dot(x_batch, self.weights1) + self.biases1)
                y_pred = self.forward(x_batch)
                self.backward(x_batch, y_batch, y_pred, learning_rate)

            y_pred_train = np.argmax(self.forward(x_train), axis=1)
            acc_train = np.mean(y_pred_train == np.argmax(y_train, axis=1))
            print('Epoch %d/%d, train accuracy: %f' % (i + 1, epochs, acc_train))

    def test(self, x_test, y_test):
        y_pred = np.argmax(self.forward(x_test), axis=1)
        acc_test = np.mean(y_pred == np.argmax(y_test, axis=1))
        print('Test accuracy: %f' % acc_test)

def shuffle_data(x, y):
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    return x[idx], y[idx]


if __name__ == '__main__':
    # 加载数据
    x_train = load_mnist_images('./data/train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('./data/train-labels-idx1-ubyte.gz')
    x_test = load_mnist_images('./data/t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('./data/t10k-labels-idx1-ubyte.gz')

    # 数据预处理.
    num_classes = 10
    x_train, y_train = preprocess_data(x_train, y_train, num_classes)
    x_test, y_test = preprocess_data(x_test, y_test, num_classes)
    x_train, y_train = shuffle_data(x_train, y_train)
    x_test, y_test = shuffle_data(x_test, y_test)
    # 初始化模型.
    model = NeuralNetwork(num_input=784, num_hidden=128, num_output=num_classes)

    # 训练模型.
    train_acc = []
    test_acc = []
    epochs = 50
    batch_size = 256
    learning_rate = 0.01
    num_samples = x_train.shape[0]

    for i in range(epochs):
        perm = np.random.permutation(num_samples)
        for j in range(0, num_samples, batch_size):
            indices = perm[j:j + batch_size]
            x_batch = x_train[indices]
            y_batch = y_train[indices]
            model.a1 = np.tanh(np.dot(x_batch, model.weights1) + model.biases1)
            y_pred = model.forward(x_batch)
            model.backward(x_batch, y_batch, y_pred, learning_rate)

        y_pred_train = np.argmax(model.forward(x_train), axis=1)
        acc_train = np.mean(y_pred_train == np.argmax(y_train, axis=1))
        train_acc.append(acc_train)

        y_pred_test = np.argmax(model.forward(x_test), axis=1)
        acc_test = np.mean(y_pred_test == np.argmax(y_test, axis=1))
        test_acc.append(acc_test)

        print('Epoch %d/%d, train accuracy: %f, test accuracy: %f' % (i + 1, epochs, acc_train, acc_test))

    # 测试模型.
    model.test(x_test, y_test)

    # 绘制结果
    plt.plot(range(1, epochs+1), train_acc, label='Training Accuracy')
    plt.plot(range(1, epochs+1), test_acc, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.show()